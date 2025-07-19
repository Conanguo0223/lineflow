import os
import ast
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import gymnasium as gym

from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HeteroConv,
    GCNConv,
    SAGEConv,
    TransformerConv,
    HGTConv,
)
from torch.nn import Linear

from lineflow.examples.waiting_time import WaitingTime
from lineflow.helpers import get_device
from lineflow.learning.helpers import make_stacked_vec_env
from lineflow.learning.curriculum import CurriculumLearningCallback
from lineflow.examples import (
    WaitingTime,
    ComplexLine,
    MultiProcess
)

import wandb
from wandb.integration.sb3 import WandbCallback


@dataclass
class PPOConfig:
    """Configuration class for PPO hyperparameters"""
    env_id: str = "WaitingTime-v0"
    exp_name: str = "ppo_hgt_experiment"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ppo_hgt_project"
    wandb_entity: Optional[str] = None
    
    # PPO hyperparameters
    num_envs: int = 1
    num_steps: int = 500
    num_minibatches: int = 4
    total_timesteps: int = 500_000
    learning_rate: float = 0.001
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_vloss: bool = True
    norm_adv: bool = True
    target_kl: Optional[float] = None
    
    # HGT specific parameters
    hidden_channels: int = 64
    out_channels: int = 4
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.1
    
    # Agent network parameters
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64
    
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps
    
    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches
    
    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size


class ImprovedHGT(torch.nn.Module):
    """Enhanced HGT with better initialization and regularization"""
    
    def __init__(self, config: PPOConfig, metadata, node_feature_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.metadata = metadata
        
        # Input projection layers with proper initialization
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:  # node_types
            self.lin_dict[node_type] = nn.Sequential(
                Linear(node_feature_dims[node_type], config.hidden_channels),
                nn.LayerNorm(config.hidden_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        
        # HGT layers with residual connections
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(config.num_layers):
            conv = HGTConv(
                config.hidden_channels, 
                config.hidden_channels, 
                metadata,
                config.num_heads
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(config.hidden_channels))
        
        # Output projection
        self.output_proj = nn.Sequential(
            Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            Linear(config.hidden_channels, config.out_channels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Input projection
        x_dict = {
            node_type: self.lin_dict[node_type](x) 
            for node_type, x in x_dict.items()
        }
        
        # HGT layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, edge_index_dict)
            # Apply residual connection and normalization
            x_dict = {
                node_type: norm(x_dict_new[node_type] + x_dict[node_type])
                for node_type in x_dict.keys()
            }
        
        return x_dict


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Improved layer initialization"""
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedPolicyAgent(nn.Module):
    """Agent that can handle multiple nodes of the same type"""

    def __init__(self, node_type: str, config: PPOConfig, action_dim: int):
        super().__init__()
        self.node_type = node_type
        self.action_dim = action_dim
        
        # Critic network - outputs a single value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(config.hidden_channels, config.critic_hidden_dim)),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            layer_init(nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim)),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            layer_init(nn.Linear(config.critic_hidden_dim, 1), std=1.0),
        )
        
        # Actor network - outputs action logits
        self.actor = nn.Sequential(
            layer_init(nn.Linear(config.hidden_channels, config.actor_hidden_dim)),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            layer_init(nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim)),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            layer_init(nn.Linear(config.actor_hidden_dim, action_dim), std=0.01),
        )
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_channels] or [batch_size * num_nodes, hidden_channels]
        Returns: [batch_size, num_nodes, 1] or [batch_size * num_nodes, 1]
        """
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, num_nodes, hidden_channels = x.shape
            x = x.reshape(-1, hidden_channels)
        
        value = self.critic(x)
        
        if len(original_shape) == 3:
            value = value.reshape(batch_size, num_nodes, 1)
        
        return value
    
    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, num_nodes, hidden_channels]
        action: [batch_size, num_nodes] if provided
        Returns: actions, log_probs, entropies, values (all with shape [batch_size, num_nodes] except values)
        """
        batch_size, num_nodes, hidden_channels = x.shape
        
        # Flatten for processing
        x_flat = x.reshape(-1, hidden_channels)
        
        # Get logits and values
        logits = self.actor(x_flat)  # [batch_size * num_nodes, action_dim]
        values = self.critic(x_flat)  # [batch_size * num_nodes, 1]
        
        # Create distributions
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()  # [batch_size * num_nodes]
            action = action.reshape(batch_size, num_nodes)
        else:
            action_flat = action.reshape(-1)
            
        # Calculate log probs and entropy
        log_probs = probs.log_prob(action.reshape(-1))  # [batch_size * num_nodes]
        entropy = probs.entropy()  # [batch_size * num_nodes]
        
        # Reshape outputs
        log_probs = log_probs.reshape(batch_size, num_nodes)
        entropy = entropy.reshape(batch_size, num_nodes)
        values = values.reshape(batch_size, num_nodes)
        
        return action, log_probs, entropy, values


class MultiAgentController:
    """Manages multiple agents with shared policies per node type"""
    
    def __init__(self, agents_dict: Dict[str, SharedPolicyAgent], node_type_mapping: Dict[int, Tuple[str, int]], device: torch.device):
        """
        agents_dict: Dictionary mapping node_type to shared policy agent
        node_type_mapping: Maps action index to (node_type, node_index)
        """
        self.agents_dict = agents_dict
        self.node_type_mapping = node_type_mapping
        self.device = device
        
        # Group action indices by node type for batch processing
        self.node_type_to_action_indices = {}
        for action_idx, (node_type, node_idx) in node_type_mapping.items():
            if node_type not in self.node_type_to_action_indices:
                self.node_type_to_action_indices[node_type] = []
            self.node_type_to_action_indices[node_type].append(action_idx)
    
    def get_actions_and_values(
        self, 
        node_features_dict: Dict[str, torch.Tensor],
        actions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and values for all nodes using shared policies
        
        Args:
            node_features_dict: Dict mapping node_type to tensor of shape [batch_size, num_nodes_of_type, hidden_channels]
            actions: Optional dict mapping node_type to actions tensor [batch_size, num_nodes_of_type]
        
        Returns:
            - combined_actions: [batch_size, total_num_actions]
            - combined_logprobs: [batch_size, total_num_actions]
            - combined_entropies: [batch_size, total_num_actions]
            - combined_values: [batch_size, total_num_actions]
        """
        batch_size = next(iter(node_features_dict.values())).shape[0]
        total_actions = len(self.node_type_mapping)
        
        # Initialize output tensors
        combined_actions = torch.zeros((batch_size, total_actions), dtype=torch.long, device=self.device)
        combined_logprobs = torch.zeros((batch_size, total_actions), device=self.device)
        combined_entropies = torch.zeros((batch_size, total_actions), device=self.device)
        combined_values = torch.zeros((batch_size, total_actions), device=self.device)
        
        # Process each node type
        for node_type, agent in self.agents_dict.items():
            if node_type not in node_features_dict:
                continue
                
            features = node_features_dict[node_type]
            action_indices = self.node_type_to_action_indices[node_type]
            
            # Get actions for this node type
            if actions is None:
                node_actions = None
            else:
                node_actions = actions[node_type]
            
            # Get outputs from shared policy
            type_actions, type_logprobs, type_entropies, type_values = agent.get_action_and_value(
                features, node_actions
            )
            
            # Place outputs in correct positions
            for i, action_idx in enumerate(action_indices):
                combined_actions[:, action_idx] = type_actions[:, i]
                combined_logprobs[:, action_idx] = type_logprobs[:, i]
                combined_entropies[:, action_idx] = type_entropies[:, i]
                combined_values[:, action_idx] = type_values[:, i]
        
        return combined_actions, combined_logprobs, combined_entropies, combined_values
    
    def get_values(self, node_features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get values for all nodes"""
        batch_size = next(iter(node_features_dict.values())).shape[0]
        total_actions = len(self.node_type_mapping)
        combined_values = torch.zeros((batch_size, total_actions), device=self.device)
        
        for node_type, agent in self.agents_dict.items():
            if node_type not in node_features_dict:
                continue
                
            features = node_features_dict[node_type]
            action_indices = self.node_type_to_action_indices[node_type]
            
            type_values = agent.get_value(features).squeeze(-1)
            
            for i, action_idx in enumerate(action_indices):
                combined_values[:, action_idx] = type_values[:, i]
        
        return combined_values


class GraphObservationProcessor:
    """Handles graph observation processing and encoding"""
    
    def __init__(
        self, 
        graph_encoder: ImprovedHGT, 
        device: torch.device, 
        target_nodes: Dict[str, List[int]],
        node_type_mapping: Dict[int, Tuple[str, int]]
    ):
        """
        target_nodes: Dict mapping node_type to list of indices to extract features from
        node_type_mapping: Maps action index to (node_type, node_index)
        """
        self.graph_encoder = graph_encoder
        self.device = device
        self.target_nodes = target_nodes
        self.node_type_mapping = node_type_mapping

    def process_observation(self, obs: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Process HeteroData observation into encoded features grouped by node type
        
        Returns:
            Dict mapping node_type to tensor of shape [batch_size, num_nodes_of_type, hidden_channels]
        """
        if not isinstance(obs, HeteroData):
            raise ValueError(f"Expected HeteroData, got {type(obs)}")
        
        obs = obs.to(self.device)
        with torch.no_grad():
            encoded_dict = self.graph_encoder(obs.x_dict, obs.edge_index_dict)
            
            features_dict = {}
            for node_type, indices in self.target_nodes.items():
                if node_type not in encoded_dict:
                    raise ValueError(f"Target node type '{node_type}' not found in encoded graph")
                
                node_features = encoded_dict[node_type]
                # Extract features for specified indices
                selected_features = []
                for idx in indices:
                    if idx >= node_features.shape[0]:
                        raise ValueError(f"Target node index {idx} out of bounds for node type '{node_type}'")
                    selected_features.append(node_features[idx])
                
                # Stack features for this node type
                if selected_features:
                    features_dict[node_type] = torch.stack(selected_features).unsqueeze(0)  # [1, num_nodes, hidden_channels]
            
            return features_dict


class PPOTrainer:
    """Main PPO trainer with shared policy support"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        
        # Setup logging
        self.run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        
        if config.track:
            wandb.init(
                project=config.wandb_project_name,
                entity=config.wandb_entity,
                sync_tensorboard=True,
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
                config=config.__dict__
            )
        
        # Setup environment
        self._setup_environment()
        
        # Setup models
        self._setup_models()
        
        # Setup storage
        self._setup_storage()
        
        # Initialize tracking variables
        self.global_step = 0
        self.start_time = time.time()
        
    def _setup_environment(self):
        """Setup environment with proper seeding"""
        # Seeding
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic
        n_cells = 5
        
        # Create environment
        line = ComplexLine(
            alternate=False,
            n_assemblies=n_cells,
            n_workers=3*n_cells,
            scrap_factor=1/n_cells,
            step_size=10,
            info=[],
            use_graph_as_states=True,
            )
        # line = MultiProcess(
        #     alternate=False,
        #     n_processes=n_cells,
        #     step_size=10,
        #     info=[('SwitchD', 'index_buffer_out')],
        #     use_graph_as_states=True,
        # )
        # line = WaitingTime(use_graph_as_states=True, step_size=10)
        
        self.envs = make_stacked_vec_env(
            line=line,
            simulation_end=4000,
            reward="parts",
            n_envs=self.config.num_envs,
            n_stack=1,
            use_graph=True,
            track_states=['S_component']
        )
        
        # Get initial state for model setup
        self.initial_state, _ = self.envs.reset()
        
    def _setup_models(self):
        """Setup graph encoder and agents with shared policies"""
        # Get node feature dimensions
        node_feature_dims = {
            node_type: self.initial_state.x_dict[node_type].shape[1]
            for node_type in self.initial_state.node_types
        }
        
        # Create graph encoder
        self.graph_encoder = ImprovedHGT(
            self.config, 
            self.initial_state.metadata(), 
            node_feature_dims
        ).to(self.device)
        
        # Get actionable nodes
        actionable_nodes = [s for s, b in zip(
            self.envs.line.state.feature_names, 
            self.envs.line.state.actionables
        ) if b]
        # special case for workerpool nodes
        
        actionable_node_names = [s.split('_')[0] for s in actionable_nodes]
        workerpool_nodes = [s for s in actionable_nodes if 'Pool' in s]
        actionable_nodes_mapped = [self.envs.line.node_mapping[t] for t in actionable_node_names]
        
        # Create mappings
        target_nodes = {}  # node_type -> list of indices
        node_type_mapping = {}  # action_idx -> (node_type, node_idx)
        node_type_action_dims = {}  # node_type -> action_dim
        
        for action_idx, (node_type, node_idx) in enumerate(actionable_nodes_mapped):
            # Build target nodes dict
            if node_type not in target_nodes:
                target_nodes[node_type] = []
            if node_idx not in target_nodes[node_type]:
                target_nodes[node_type].append(node_idx)
            
            # Build node type mapping
            node_type_mapping[action_idx] = (node_type, node_idx)
            
            # Track action dimensions per node type
            if node_type not in node_type_action_dims:
                node_type_action_dims[node_type] = self.envs.action_space.nvec[action_idx]
        
        self.target_nodes = target_nodes
        self.node_type_mapping = node_type_mapping
        
        # Setup observation processor
        self.obs_processor = GraphObservationProcessor(
            self.graph_encoder, 
            self.device,
            target_nodes=target_nodes,
            node_type_mapping=node_type_mapping
        )
        
        # Create shared policy agents (one per node type)
        agents_dict = {}
        for node_type, action_dim in node_type_action_dims.items():
            agent = SharedPolicyAgent(
                node_type=node_type,
                config=self.config,
                action_dim=action_dim
            ).to(self.device)
            agents_dict[node_type] = agent
            print(f"Created shared policy for node type '{node_type}' with action dim {action_dim}")
        
        self.multi_agent_controller = MultiAgentController(agents_dict, node_type_mapping, self.device)
        
        # Setup optimizer with all networks
        all_params = list(self.graph_encoder.parameters())
        for agent in agents_dict.values():
            all_params += list(agent.parameters())
        
        self.optimizer = optim.Adam(all_params, lr=self.config.learning_rate, eps=1e-5)
        
        # Setup learning rate scheduler
        if self.config.anneal_lr:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=1.0, 
                end_factor=0.0, 
                total_iters=self.config.num_iterations
            )
    
    def _setup_storage(self):
        """Setup storage buffers for shared policies"""
        total_actions = len(self.node_type_mapping)
        
        # Storage for grouped features by node type
        self.obs = {}
        for node_type, indices in self.target_nodes.items():
            self.obs[node_type] = torch.zeros(
                (self.config.num_steps, self.config.num_envs, len(indices), self.config.hidden_channels)
            ).to(self.device)
        
        # Storage for combined outputs
        self.actions = torch.zeros((self.config.num_steps, self.config.num_envs, total_actions), dtype=torch.long).to(self.device)
        self.logprobs = torch.zeros((self.config.num_steps, self.config.num_envs, total_actions)).to(self.device)
        self.rewards = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        self.dones = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        self.values = torch.zeros((self.config.num_steps, self.config.num_envs, total_actions)).to(self.device)

    def train(self):
        """Main training loop"""
        # Initialize environment
        next_obs, _ = self.envs.reset(seed=self.config.seed)
        next_obs_encoded = self.obs_processor.process_observation(next_obs)
        next_done = torch.zeros(self.config.num_envs).to(self.device)
        
        # Training metrics
        episode_reward = 0
        episode_length = 0
        
        for iteration in range(1, self.config.num_iterations + 1):
            # Update learning rate
            if self.config.anneal_lr:
                self.scheduler.step()
            
            # Collect rollout
            for step in range(self.config.num_steps):
                self.global_step += self.config.num_envs
                
                # Store observations
                for node_type, features in next_obs_encoded.items():
                    self.obs[node_type][step] = features
                self.dones[step] = next_done
                
                # Get actions from shared policies
                with torch.no_grad():
                    actions, logprobs, _, values = \
                        self.multi_agent_controller.get_actions_and_values(next_obs_encoded)
                    
                    self.actions[step] = actions
                    self.logprobs[step] = logprobs
                    self.values[step] = values
                
                # Environment step
                next_obs_raw, reward, terminations, truncations, infos = self.envs.step(actions.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                
                # Update metrics
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Process next observation
                next_obs_encoded = self.obs_processor.process_observation(next_obs_raw)
                next_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
                
                # Handle episode termination
                if next_done.any():
                    print(f"Episode finished - Reward: {episode_reward:.2f}, Length: {episode_length}")
                    self.writer.add_scalar("charts/episodic_return", episode_reward, self.global_step)
                    self.writer.add_scalar("charts/episodic_length", episode_length, self.global_step)
                    episode_reward = 0
                    episode_length = 0
                    
                    # Reset the environment
                    reset_obs, _ = self.envs.reset()
                    next_obs_encoded = self.obs_processor.process_observation(reset_obs)
            
            # Update all policies
            self._update_policies(next_obs_encoded, next_done)
            
            # Log performance
            sps = int(self.global_step / (time.time() - self.start_time))
            print(f"Iteration {iteration}, SPS: {sps}")
            self.writer.add_scalar("charts/SPS", sps, self.global_step)
    
    def _update_policies(self, next_obs: Dict[str, torch.Tensor], next_done: torch.Tensor):
        """Update shared policies using PPO"""
        # Get next values
        with torch.no_grad():
            next_values = self.multi_agent_controller.get_values(next_obs)
        
        # Calculate advantages (shared across all actions for now)
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        
        for t in reversed(range(self.config.num_steps)):
            if t == self.config.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                # Average values across all actions for global advantage
                nextvalues = next_values.mean(dim=1)
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1].mean(dim=1)
            
            current_values = self.values[t].mean(dim=1)
            delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - current_values
            advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
        
        # Expand advantages to all actions
        advantages_expanded = advantages.unsqueeze(-1).expand_as(self.values)
        returns = advantages_expanded + self.values
        
        # Flatten batches
        b_obs = {}
        for node_type, obs in self.obs.items():
            b_obs[node_type] = obs.reshape((-1, len(self.target_nodes[node_type]), self.config.hidden_channels))
        
        b_actions = {}
        for node_type in self.target_nodes.keys():
            # Gather actions for this node type
            action_indices = self.multi_agent_controller.node_type_to_action_indices[node_type]
            node_actions = []
            for step in range(self.config.num_steps):
                for env in range(self.config.num_envs):
                    env_actions = []
                    for idx in action_indices:
                        env_actions.append(self.actions[step, env, idx])
                    node_actions.append(torch.stack(env_actions))
            b_actions[node_type] = torch.stack(node_actions)
        
        b_logprobs = self.logprobs.reshape(-1, len(self.node_type_mapping))
        b_advantages = advantages_expanded.reshape(-1, len(self.node_type_mapping))
        b_returns = returns.reshape(-1, len(self.node_type_mapping))
        b_values = self.values.reshape(-1, len(self.node_type_mapping))
        
        # Optimize policies
        b_inds = np.arange(self.config.batch_size)
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Prepare minibatch data
                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                mb_actions = {k: v[mb_inds] for k, v in b_actions.items()}
                
                # Forward pass
                _, newlogprobs, entropies, newvalues = \
                    self.multi_agent_controller.get_actions_and_values(mb_obs, mb_actions)
                
                # Calculate losses
                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalues_flat = newvalues.view(-1)
                b_returns_flat = b_returns[mb_inds].view(-1)
                b_values_flat = b_values[mb_inds].view(-1)
                
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalues_flat - b_returns_flat) ** 2
                    v_clipped = b_values_flat + torch.clamp(
                        newvalues_flat - b_values_flat,
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_flat) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalues_flat - b_returns_flat) ** 2).mean()
                
                entropy_loss = entropies.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients for all parameters
                all_params = list(self.graph_encoder.parameters())
                for agent in self.multi_agent_controller.agents_dict.values():
                    all_params += list(agent.parameters())
                
                nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
                self.optimizer.step()
            
            # Early stopping based on KL divergence
            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break
        
        # Log metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        
        # Log per-node-type metrics
        for node_type in self.multi_agent_controller.agents_dict.keys():
            num_nodes = len(self.target_nodes[node_type])
            self.writer.add_scalar(f"info/{node_type}/num_nodes", num_nodes, self.global_step)
    
    def cleanup(self):
        """Cleanup resources"""
        self.envs.close()
        self.writer.close()
        if self.config.track:
            wandb.finish()


def main():
    """Main function to run training"""
    config = PPOConfig()
    trainer = PPOTrainer(config)
    
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()