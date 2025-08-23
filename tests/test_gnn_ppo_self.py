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
from typing import Dict, Any, Optional, Tuple

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
    HANConv,
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
            conv = HANConv(
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

def group_indexes_by_type(node_types_list):
    """
    Groups indexes by node type from a list of node type names.
    
    Args:
        node_types_list: List of node type strings
    
    Returns:
        Dictionary mapping node type to list of indexes
    """
    type_indexes = {}
    
    for index, node_type in enumerate(node_types_list):
        if node_type not in type_indexes:
            type_indexes[node_type] = []
        type_indexes[node_type].append(index)
    
    return type_indexes

class ImprovedAgent(nn.Module):
    """Enhanced agent with better architecture and regularization"""

    def __init__(self, config: PPOConfig, action_dim: int):
        super().__init__()
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(config.hidden_channels, config.critic_hidden_dim)),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            layer_init(nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim)),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            layer_init(nn.Linear(config.critic_hidden_dim, 1), std=1.0),
        )
        
        # Actor network
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
        return self.critic(x)
    
    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class GraphObservationProcessor:
    """Handles graph observation processing and encoding"""
    
    def __init__(
        self, 
        graph_encoder: ImprovedHGT, 
        device: torch.device, 
        target_nodes: Optional[Dict[str, list]] = None
    ):
        """
        target_nodes: Dict mapping node_type to list of indices to extract features from.
        Example: {'Switch': [1, 2], 'Buffer': [0]}
        """
        self.graph_encoder = graph_encoder
        self.device = device
        self.target_nodes = target_nodes if target_nodes is not None else {'Switch': [1]}

    def process_observation(self, obs: HeteroData) -> torch.Tensor:
        """Process HeteroData observation into encoded features for multiple node types and indices"""
        if not isinstance(obs, HeteroData):
            raise ValueError(f"Expected HeteroData, got {type(obs)}")
        
        obs = obs.to(self.device)
        with torch.no_grad():
            encoded_dict = self.graph_encoder(obs.x_dict, obs.edge_index_dict)
            features = []
            for node_type, indices in self.target_nodes.items():
                if node_type not in encoded_dict:
                    raise ValueError(f"Target node type '{node_type}' not found in encoded graph")
                node_features = encoded_dict[node_type]
                for idx in indices:
                    if idx >= node_features.shape[0]:
                        raise ValueError(f"Target node index {idx} out of bounds for node type '{node_type}'")
                    features.append(node_features[idx])
            # Stack all selected node features into a single tensor
            return torch.stack(features)


class PPOTrainer:
    """Main PPO trainer with improved organization"""
    
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
            track_states=['S_component']
        )
        
        # Get initial state for model setup
        self.initial_state, _ = self.envs.reset()
        
    def _setup_models(self):
        """Setup graph encoder and agent"""
        # Get node feature dimensions
        node_feature_dims = {
            node_type: self.initial_state.x_dict[node_type].shape[1]
            for node_type in self.initial_state.node_types
        }
        
        # Create models
        self.graph_encoder = ImprovedHGT(
            self.config, 
            self.initial_state.metadata(), 
            node_feature_dims
        ).to(self.device)
        
        self.agent = ImprovedAgent(
            config = self.config, 
            action_dim = self.envs.action_space.nvec[0]
        ).to(self.device)
        actionable_nodes = [s for s, b in zip(self.envs.line.state.feature_names, self.envs.line.state.actionables) if b]
        actionable_node_names = [s.split('_')[0] for s in actionable_nodes]
        actionable_nodes = [self.envs.line.node_mapping[t] for t in actionable_node_names]
        
        # Group indices by node type
        target_nodes = {}
        for node_type, index in actionable_nodes:
            if node_type not in target_nodes:
                target_nodes[node_type] = []
            if index not in target_nodes[node_type]:
                target_nodes[node_type].append(index)
        # Setup observation processor
        self.obs_processor = GraphObservationProcessor(
            self.graph_encoder, 
            self.device,
            # target_nodes={'Switch': [0,1]}  # Example target nodes
            target_nodes=target_nodes  # Only process 'S_component' node type
        )
        target_nodes = group_indexes_by_type(actionable_node_names)
        agent_list = []
        # TODO: make this smarter, currently only uses the first action dim for the agents, might have different types?
        for node_type in target_nodes.keys():
            agent_list.append(ImprovedAgent(
                config=self.config,
                action_dim=self.envs.action_space.nvec[target_nodes[node_type][0]]
            ).to(self.device))
        # Setup optimizer with both networks
        self.agent_list = agent_list
        all_params = []
        for agent in self.agent_list:
            all_params += list(agent.parameters())
        all_params += list(self.graph_encoder.parameters())
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
        """Setup storage buffers"""
        # If your obs have multiple node features (e.g., for two target nodes), 
        # obs shape should be (num_steps, num_envs, num_target_nodes, hidden_channels)
        num_target_nodes = sum(len(indices) for indices in self.obs_processor.target_nodes.values())
        self.obs = torch.zeros(
            (self.config.num_steps, self.config.num_envs, num_target_nodes, self.config.hidden_channels)
        ).to(self.device)
        self.actions = torch.zeros((self.config.num_steps, self.config.num_envs, num_target_nodes), dtype=torch.long).to(self.device)
        self.logprobs = torch.zeros((self.config.num_steps, self.config.num_envs, num_target_nodes)).to(self.device)
        self.rewards = torch.zeros((self.config.num_steps, self.config.num_envs, num_target_nodes)).to(self.device)
        self.dones = torch.zeros((self.config.num_steps, self.config.num_envs, num_target_nodes)).to(self.device)
        self.values = torch.zeros((self.config.num_steps, self.config.num_envs, num_target_nodes)).to(self.device)

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
                self.obs[step] = next_obs_encoded
                self.dones[step] = next_done
                
                # Get action
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs_encoded)
                    self.values[step] = value.flatten()
                
                self.actions[step] = action
                self.logprobs[step] = logprob
                
                # Environment step
                next_obs_raw, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                
                # Update metrics
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Process next observation
                next_obs_encoded = self.obs_processor.process_observation(next_obs_raw)
                next_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
                
                # Handle episode termination
                # Handle episode termination
                if next_done.any():
                    print(f"Episode finished - Reward: {episode_reward:.2f}, Length: {episode_length}")
                    self.writer.add_scalar("charts/episodic_return", episode_reward, self.global_step)
                    self.writer.add_scalar("charts/episodic_length", episode_length, self.global_step)
                    episode_reward = 0
                    episode_length = 0
                    
                    # ✅ ADD: Reset the environment
                    reset_obs, _ = self.envs.reset()
                    next_obs_encoded = self.obs_processor.process_observation(reset_obs)
            # Update policy
            self._update_policy(next_obs_encoded, next_done)
            
            # Log performance
            sps = int(self.global_step / (time.time() - self.start_time))
            print(f"Iteration {iteration}, SPS: {sps}")
            self.writer.add_scalar("charts/SPS", sps, self.global_step)
    
    def _update_policy(self, next_obs: torch.Tensor, next_done: torch.Tensor):
        """Update policy using PPO"""
        # Calculate advantages
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + self.values
        
        # Flatten batch
        b_obs = self.obs.reshape((-1, self.config.hidden_channels))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # Optimize policy
        b_inds = np.arange(self.config.batch_size)
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Forward pass
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                # Calculate losses
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent.parameters()) + list(self.graph_encoder.parameters()), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            # Early stopping
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