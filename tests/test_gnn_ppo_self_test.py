import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, Linear, HGTConv, HANConv
from lineflow.learning.helpers import make_stacked_vec_env
from lineflow.examples import (
    WaitingTime,
    ComplexLine,
    MultiProcess
)

import numpy as np
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import os

# GPU Setup and Detection
def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set memory allocation strategy for better performance
        torch.backends.cudnn.benchmark = True
        
        # Clear cache
        torch.cuda.empty_cache()
        
    elif torch.backends.mps.is_available():  # For Apple Silicon
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

# Global device variable
DEVICE = setup_device()

def move_hetero_data_to_device(hetero_data, device):
    """Move HeteroData to specified device"""
    if hetero_data is None:
        return None
    
    # Create new HeteroData on the target device
    new_data = HeteroData()
    
    # Move node features
    for node_type, x in hetero_data.x_dict.items():
        new_data[node_type].x = x.to(device) if torch.is_tensor(x) else x
    
    # Move edge indices
    for edge_type, edge_index in hetero_data.edge_index_dict.items():
        new_data[edge_type].edge_index = edge_index.to(device) if torch.is_tensor(edge_index) else edge_index
    
    # Move edge attributes if they exist
    if hasattr(hetero_data, 'edge_attr_dict'):
        for edge_type, edge_attr in hetero_data.edge_attr_dict.items():
            new_data[edge_type].edge_attr = edge_attr.to(device) if torch.is_tensor(edge_attr) else edge_attr
    
    return new_data




class HGT_policy(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, 
                 action_dims, data, device=None):
        super().__init__()
        
        # Set device
        self.device = device if device is not None else DEVICE
        
        # Store node types and metadata
        self.node_types = list(data.node_types)
        self.hidden_channels = hidden_channels
        self.action_dims = action_dims
        self.num_actions = len(action_dims)

        # Project input features to unified hidden dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        # Heterogeneous graph convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, dropout=0.1)  # Added dropout for regularization
            self.convs.append(conv)

        # Shared feature extractor
        total_features = hidden_channels * len(self.node_types)
        self.shared_layers = nn.Sequential(
            nn.Linear(total_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)   # Added dropout
        )
        
        # Separate policy heads for each action component
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, action_dim)
            ) for action_dim in action_dims
        ])
        
        # Single value head (shared across all actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Move model to device
        self.to(self.device)

    def forward(self, hetero_data):
        """Forward pass with GPU support"""
        # Ensure data is on the correct device
        if not self._is_data_on_device(hetero_data):
            hetero_data = move_hetero_data_to_device(hetero_data, self.device)
        
        x_dict = hetero_data.x_dict
        edge_index_dict = hetero_data.edge_index_dict
        
        # Project input features to hidden dimension
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Apply HAN convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # POOLING STAGE: Aggregate node features by type
        pooled_features = []
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                # Mean pooling over all nodes of this type
                pooled = torch.mean(x_dict[node_type], dim=0, keepdim=True)
                pooled_features.append(pooled)
            else:
                # Handle missing node types with zero features
                zero_features = torch.zeros(1, self.hidden_channels, device=self.device)
                pooled_features.append(zero_features)
        
        # Concatenate all pooled node type features
        global_features = torch.cat(pooled_features, dim=1)
        
        # Apply shared layers
        shared_features = self.shared_layers(global_features)
        
        # Get separate action logits for each component
        action_logits = []
        for policy_head in self.policy_heads:
            logits = policy_head(shared_features)
            action_logits.append(logits)
        
        # Get value
        value = self.value_head(shared_features)
        
        return action_logits, value
    
    def _is_data_on_device(self, hetero_data):
        """Check if HeteroData is on the correct device"""
        if not hetero_data.x_dict:
            return True
        
        # Check first tensor
        first_tensor = next(iter(hetero_data.x_dict.values()))
        return first_tensor.device == self.device

class PPOAgent:
    """GPU-optimized PPO Agent for MultiDiscrete HeteroData environments"""

    def __init__(self, sample_state, action_dims, lr=1e-3, gamma=0.99,
                 eps_clip=0.2, k_epochs=4, hidden_dim=64, batch_size=64,
                 buffer_size=2048, device=None):
        
        # Set device
        self.device = device if device is not None else DEVICE
        print(f"PPO Agent using device: {self.device}")
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dims = action_dims
        self.num_actions = len(action_dims)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Initialize policy network
        self.policy = HGT_policy(
            hidden_channels=hidden_dim,
            out_channels=None,
            num_heads=4,
            num_layers=2,
            action_dims=action_dims,
            data=sample_state,
            device=self.device
        )
        
        # Use different learning rates and optimizers for better GPU performance
        if self.device.type == 'cuda':
            # Use AdamW with weight decay for better generalization on GPU
            self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = PPOBuffer(buffer_size)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Move sample state to device for future use
        self.sample_state_device = move_hetero_data_to_device(sample_state, self.device)
    
    def select_action(self, state, deterministic=False):
        """GPU-optimized action selection"""
        # Move state to device if needed
        if not self._is_state_on_device(state):
            state = move_hetero_data_to_device(state, self.device)
        
        with torch.no_grad():
            action_logits_list, value = self.policy(state)
            
        # Handle different action space scenarios
        if len(self.action_dims) == 1:
            # Single discrete action
            logits = action_logits_list[0]
            dist = Categorical(logits=logits)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.squeeze().item()
        
        else:
            # Multi-discrete actions
            actions = []
            log_probs = []
            
            for i, logits in enumerate(action_logits_list):
                dist = Categorical(logits=logits)
                
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action)
                
                actions.append(action.item())
                log_probs.append(log_prob.item())
            
            return actions, log_probs, value.squeeze().item()
    
    def evaluate_actions(self, state, actions):
        """GPU-optimized action evaluation"""
        # Ensure state is on device
        if not self._is_state_on_device(state):
            state = move_hetero_data_to_device(state, self.device)
        
        action_logits_list, value = self.policy(state)
        
        total_log_prob = 0
        total_entropy = 0
        
        # Handle single vs multi-discrete actions
        if len(self.action_dims) == 1:
            logits = action_logits_list[0]
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_prob, entropy, value.squeeze()
        else:
            for i, logits in enumerate(action_logits_list):
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(actions[i])
                entropy = dist.entropy()
                
                total_log_prob += log_prob
                total_entropy += entropy
            
            return total_log_prob, total_entropy, value.squeeze()
    
    def should_update(self):
        """Check if buffer is full enough for update"""
        return len(self.buffer.states) >= self.buffer_size
    
    def update(self):
        """GPU-optimized PPO update with mixed precision support"""
        if len(self.buffer.states) < self.batch_size:
            return
        
        # Use automatic mixed precision for faster training on modern GPUs
        use_amp = self.device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Get all experiences
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get_batch()
        
        # Compute advantages and returns for all experiences
        advantages, returns = self.compute_gae_batch(rewards, values, dones)
        
        # Convert to tensors and move to device
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Handle action tensors based on action space type
        if len(self.action_dims) == 1:
            old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        else:
            old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create indices for batch sampling
        indices = np.arange(len(states))
        
        # PPO update with mini-batches
        for epoch in range(self.k_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                if use_amp:
                    # Mixed precision forward pass
                    with torch.cuda.amp.autocast():
                        batch_loss = self.compute_batch_loss_clean(
                            states, actions_tensor, old_log_probs_tensor,
                            advantages, returns, batch_indices
                        )
                    
                    # Mixed precision backward pass
                    scaler.scale(batch_loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Regular precision
                    batch_loss = self.compute_batch_loss_clean(
                        states, actions_tensor, old_log_probs_tensor,
                        advantages, returns, batch_indices
                    )
                    
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        # Clear GPU cache periodically
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def compute_batch_loss_clean(self, states, actions_tensor, old_log_probs_tensor, 
                                advantages, returns, batch_indices):
        """GPU-optimized batch loss computation"""
        batch_policy_loss = 0
        batch_value_loss = 0
        batch_entropy = 0
        
        for idx in batch_indices:
            # Get current policy outputs
            if len(self.action_dims) == 1:
                log_prob, entropy, value = self.evaluate_actions(states[idx], actions_tensor[idx])
                old_log_prob = old_log_probs_tensor[idx]
            else:
                log_prob, entropy, value = self.evaluate_actions(states[idx], actions_tensor[idx])
                old_log_prob = old_log_probs_tensor[idx].sum()
            
            # Compute ratio
            ratio = torch.exp(log_prob - old_log_prob)
            
            # Compute surrogate losses
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
            
            # Accumulate losses
            batch_policy_loss += -torch.min(surr1, surr2)
            batch_value_loss += self.mse_loss(value, returns[idx])
            batch_entropy += entropy
        
        # Average over batch
        batch_size = len(batch_indices)
        total_loss = (batch_policy_loss / batch_size + 
                     0.5 * batch_value_loss / batch_size - 
                     0.01 * batch_entropy / batch_size)
        
        return total_loss
    
    def compute_gae_batch(self, rewards, values, dones, lam=0.95):
        """GAE computation (CPU-based as it's sequential)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1] * (1 - dones[i])
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def _is_state_on_device(self, state):
        """Check if state is on the correct device"""
        if hasattr(state, 'x_dict') and state.x_dict:
            first_tensor = next(iter(state.x_dict.values()))
            return first_tensor.device == self.device
        return True
    
# Keep PPOBuffer the same as it handles CPU data
class PPOBuffer:
    """Experience buffer (CPU-based for efficiency)"""
    
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        # Remove oldest if buffer is full
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
        
        # Store state on CPU to save GPU memory
        if hasattr(state, 'cpu'):
            state = state.cpu()
        elif hasattr(state, 'x_dict'):
            # Move HeteroData to CPU
            cpu_state = HeteroData()
            for node_type, x in state.x_dict.items():
                cpu_state[node_type].x = x.cpu() if torch.is_tensor(x) else x
            for edge_type, edge_index in state.edge_index_dict.items():
                cpu_state[edge_type].edge_index = edge_index.cpu() if torch.is_tensor(edge_index) else edge_index
            state = cpu_state
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get_batch(self):
        return (self.states, self.actions, self.rewards, 
                self.values, self.log_probs, self.dones)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {max_memory:.2f}GB")
        return memory_allocated, memory_reserved, max_memory
    return 0, 0, 0

def train_ppo_optimized(env, agent, num_episodes=1000, max_steps=200, 
                       update_frequency=10, save_frequency=100):
    """CLEANER training loop"""
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action - CLEANER HANDLING
            action_result = agent.select_action(state)
            
            # Multi actions: action_result = (actions_list, log_probs_list, value)
            actions, log_probs, value = action_result
            if isinstance(actions, list):
                action_for_env = np.array(actions)
            else:
                action_for_env = np.array([actions])
            log_probs_to_store = log_probs
            
            # Take action in environment
            step_result = env.step(action_for_env)
            
            # Handle different step return formats
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            elif len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                raise ValueError(f"Unexpected step result: {len(step_result)} elements")
            
            agent.buffer.store(state, actions, reward, value, log_probs_to_store, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update policy when buffer is full or every update_frequency episodes
        if agent.should_update() or (episode + 1) % update_frequency == 0:
            update_start = time.time()
            agent.update()
            update_time = time.time() - update_start
            if episode % save_frequency == 0 and episode > 0:
                print(f"  Update took {update_time:.2f}s")
        
        # Progress reporting
        if episode % save_frequency == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-save_frequency:])
            avg_length = np.mean(episode_lengths[-save_frequency:])
            elapsed_time = time.time() - start_time
            episodes_per_sec = episode / elapsed_time
            
            print(f"Episode {episode}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Episodes/sec: {episodes_per_sec:.2f}")
            print(f"  Buffer size: {len(agent.buffer.states)}")
            print(f"  Time elapsed: {elapsed_time:.1f}s")
            
            # Save intermediate results
            np.save(f"ppo_gnn_rewards_intermediate_{episode}.npy", np.array(episode_rewards))

    return np.array(episode_rewards)

# Update your training functions to include GPU monitoring
def train_ppo_with_evaluation_gpu(env, agent, num_episodes=1000, max_steps=200, 
                                 update_frequency=10, save_frequency=50,
                                 eval_frequency=100, num_eval_episodes=5,
                                 save_dir="ppo_training_results"):
    """GPU-optimized training loop with memory monitoring"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    eval_results_history = []
    start_time = time.time()
    
    eval_env = env
    
    print(f"Starting GPU-optimized PPO training...")
    print(f"Device: {agent.device}")
    print(f"Results will be saved to: {save_dir}")
    
    # Initial GPU memory check
    if agent.device.type == 'cuda':
        monitor_gpu_usage()
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action (automatically handles device transfer)
            action_result = agent.select_action(state)
            
            # Handle action format
            actions, log_probs, value = action_result
            if isinstance(actions, list):
                action_for_env = np.array(actions)
            else:
                action_for_env = np.array([actions])
            
            # Take action
            step_result = env.step(action_for_env)
            
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            elif len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            
            # Store experience (buffer handles CPU storage)
            agent.buffer.store(state, actions, reward, value, log_probs, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update policy (GPU-optimized)
        if agent.should_update() or (episode + 1) % update_frequency == 0:
            update_start = time.time()
            agent.update()
            update_time = time.time() - update_start
            
            if episode % save_frequency == 0 and episode > 0:
                print(f"  GPU Update took {update_time:.2f}s")
                if agent.device.type == 'cuda':
                    monitor_gpu_usage()
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0 or episode == num_episodes - 1:
            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            eval_stats = evaluate_agent(
                agent, eval_env, 
                num_eval_episodes=num_eval_episodes,
                max_steps=max_steps,
                deterministic=True
            )
            
            eval_results_history.append({
                'episode': episode + 1,
                'stats': eval_stats
            })
            
            # Save intermediate results
            save_agent(agent, os.path.join(save_dir, f"agent_ep_{episode + 1}.pth"))
            
            # Monitor GPU after evaluation
            if agent.device.type == 'cuda':
                monitor_gpu_usage()
        
        # Progress reporting
        if episode % save_frequency == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-save_frequency:])
            avg_length = np.mean(episode_lengths[-save_frequency:])
            elapsed_time = time.time() - start_time
            episodes_per_sec = episode / elapsed_time
            
            print(f"\nEpisode {episode}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Episodes/sec: {episodes_per_sec:.2f}")
            print(f"  Buffer size: {len(agent.buffer.states)}")
            print(f"  Time elapsed: {elapsed_time:.1f}s")
    
    # Final cleanup
    if agent.device.type == 'cuda':
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    return episode_rewards, eval_results_history


def test_fast_training():
    """Quick test to verify everything works - WITH BETTER ACTION HANDLING"""
    print("Running fast test...")
    
    # Small environment for testing
    # line = WaitingTime(
    #     info=[],
    #     processing_time_source=5,
    #     use_graph_as_states=True
    # )
    # line_name = "ComplexLine"
    n_cells = 3
    line = ComplexLine(
        alternate=False,
            n_assemblies=n_cells,
            n_workers=3*n_cells,
            scrap_factor=0,
            step_size=10,
            info=[],
            use_graph_as_states=True
            )
    env = make_stacked_vec_env(
        line=line,
        simulation_end=100,
        reward="parts",
        n_envs=1,
        n_stack=1,
        track_states=['S_component']
    )
    
    sample_state = line._graph_states
    
    # Check action space and print details
    print(f"Environment action space: {env.action_space}")
    print(f"Action space type: {type(env.action_space)}")
    
    if hasattr(env.action_space, 'nvec'):
        action_dims = env.action_space.nvec.tolist()
        print(f"MultiDiscrete action dims: {action_dims}")
    elif hasattr(env.action_space, 'n'):
        action_dims = [env.action_space.n]
        print(f"Single Discrete action dim: {action_dims}")
    else:
        action_dims = [4]
        print(f"Fallback action dim: {action_dims}")
    
    # Small agent for testing
    agent = PPOAgent(
        sample_state=sample_state,
        action_dims=action_dims,
        lr=1e-3,
        hidden_dim=32,
        batch_size=16,
        buffer_size=64
    )
    
    # Test action selection first
    print("Testing action selection...")
    action_result = agent.select_action(sample_state)
    print(f"Action result: {action_result}")
    print(f"Action result type: {type(action_result)}")
    
    # Quick training
    start_time = time.time()
    rewards = train_ppo_optimized(
        env, agent, 
        num_episodes=20,  # Even fewer for quick test
        max_steps=50,
        update_frequency=5,
        save_frequency=10
    )
    
    test_time = time.time() - start_time
    print(f"Test completed in {test_time:.2f}s")
    print(f"Final average reward: {np.mean(rewards[-5:]):.2f}")
    
    return rewards
def evaluate_agent(agent, eval_env, num_eval_episodes=10, max_steps=1000, 
                  deterministic=True, render=False):
    """
    Evaluate the agent's performance over multiple episodes
    
    Args:
        agent: Trained PPO agent
        eval_env: Environment for evaluation
        num_eval_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic policy
        render: Whether to render episodes (if supported)
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating agent over {num_eval_episodes} episodes...")
    
    eval_rewards = []
    eval_lengths = []
    eval_success_rate = 0
    detailed_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_details': []
    }
    
    agent.policy.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        for episode in range(num_eval_episodes):
            state = eval_env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            episode_length = 0
            episode_info = []
            
            for step in range(max_steps):
                # Use deterministic policy for evaluation
                action_result = agent.select_action(state, deterministic=deterministic)
                
                # Handle action format
                actions, log_probs, value = action_result
                if isinstance(actions, list):
                    action_for_env = np.array(actions)
                else:
                    action_for_env = np.array([actions])
                
                # Take action
                step_result = eval_env.step(action_for_env)
                
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                elif len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                
                episode_reward += reward
                episode_length += 1
                episode_info.append({
                    'step': step,
                    'reward': reward,
                    'action': actions if isinstance(actions, list) else [actions],
                    'info': info
                })
                
                if render and hasattr(eval_env, 'render'):
                    eval_env.render()
                
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            # Check for success (you can customize this based on your environment)
            if episode_reward > 0:  # Simple success criterion
                eval_success_rate += 1
            
            detailed_stats['episode_rewards'].append(episode_reward)
            detailed_stats['episode_lengths'].append(episode_length)
            detailed_stats['episode_details'].append(episode_info)
            
            print(f"  Eval Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    agent.policy.train()  # Set back to training mode
    
    # Compute statistics
    eval_stats = {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'min_reward': np.min(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'mean_length': np.mean(eval_lengths),
        'std_length': np.std(eval_lengths),
        'success_rate': eval_success_rate / num_eval_episodes,
        'all_rewards': eval_rewards,
        'all_lengths': eval_lengths,
        'detailed_stats': detailed_stats
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean Length: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}")
    print(f"  Success Rate: {eval_stats['success_rate']:.2%}")
    print(f"  Reward Range: [{eval_stats['min_reward']:.2f}, {eval_stats['max_reward']:.2f}]")
    
    return eval_stats

def save_agent(agent, save_path="ppo_agent.pth", include_optimizer=True):
    """
    Save the trained agent
    
    Args:
        agent: PPO agent to save
        save_path: Path to save the agent
        include_optimizer: Whether to save optimizer state
    """
    save_dict = {
        'policy_state_dict': agent.policy.state_dict(),
        'action_dims': agent.action_dims,
        'gamma': agent.gamma,
        'eps_clip': agent.eps_clip,
        'k_epochs': agent.k_epochs,
        'batch_size': agent.batch_size,
        'buffer_size': agent.buffer_size
    }
    
    if include_optimizer:
        save_dict['optimizer_state_dict'] = agent.optimizer.state_dict()
    
    torch.save(save_dict, save_path)
    print(f"Agent saved to {save_path}")

def load_agent(save_path, sample_state, device='cpu'):
    """
    Load a saved agent
    
    Args:
        save_path: Path to saved agent
        sample_state: Sample state for initializing the policy
        device: Device to load the agent on
    
    Returns:
        PPOAgent: Loaded agent
    """
    checkpoint = torch.load(save_path, map_location=device)
    
    agent = PPOAgent(
        sample_state=sample_state,
        action_dims=checkpoint['action_dims'],
        gamma=checkpoint['gamma'],
        eps_clip=checkpoint['eps_clip'],
        k_epochs=checkpoint['k_epochs'],
        batch_size=checkpoint['batch_size'],
        buffer_size=checkpoint['buffer_size']
    )
    
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Agent loaded from {save_path}")
    return agent

def plot_training_progress(training_rewards, eval_results_history, save_path="training_progress.png"):
    """
    Plot training progress including evaluation results
    
    Args:
        training_rewards: Array of training episode rewards
        eval_results_history: List of evaluation results over training
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training rewards
    axes[0, 0].plot(training_rewards, alpha=0.3, color='blue', label='Episode Rewards')
    
    # Moving average
    window_size = min(100, len(training_rewards) // 10)
    if len(training_rewards) >= window_size:
        moving_avg = np.convolve(training_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size-1, len(training_rewards)), moving_avg, 
                       color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Evaluation rewards over time
    if eval_results_history:
        eval_episodes = [result['episode'] for result in eval_results_history]
        eval_means = [result['stats']['mean_reward'] for result in eval_results_history]
        eval_stds = [result['stats']['std_reward'] for result in eval_results_history]
        
        axes[0, 1].errorbar(eval_episodes, eval_means, yerr=eval_stds, 
                           marker='o', capsize=5, capthick=2, label='Eval Mean ± Std')
        axes[0, 1].set_xlabel('Training Episode')
        axes[0, 1].set_ylabel('Evaluation Reward')
        axes[0, 1].set_title('Evaluation Performance Over Training')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Episode lengths
    if eval_results_history:
        eval_lengths = [result['stats']['mean_length'] for result in eval_results_history]
        axes[1, 0].plot(eval_episodes, eval_lengths, marker='s', color='green', label='Eval Mean Length')
        axes[1, 0].set_xlabel('Training Episode')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].set_title('Episode Lengths Over Training')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Success rate
    if eval_results_history:
        success_rates = [result['stats']['success_rate'] for result in eval_results_history]
        axes[1, 1].plot(eval_episodes, success_rates, marker='^', color='purple', label='Success Rate')
        axes[1, 1].set_xlabel('Training Episode')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate Over Training')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved to {save_path}")
    return fig

def train_ppo_with_evaluation(env, agent, num_episodes=1000, max_steps=200, 
                             update_frequency=10, save_frequency=50,
                             eval_frequency=100, num_eval_episodes=5,
                             save_dir="ppo_training_results"):
    """
    Enhanced training loop with evaluation and saving
    
    Args:
        env: Training environment
        agent: PPO agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_frequency: How often to update the policy
        save_frequency: How often to print progress
        eval_frequency: How often to evaluate the agent
        num_eval_episodes: Number of episodes for each evaluation
        save_dir: Directory to save results
    
    Returns:
        tuple: (training_rewards, eval_results_history)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    eval_results_history = []
    start_time = time.time()
    
    # Create evaluation environment (same as training for now)
    eval_env = env  # You might want to create a separate eval environment
    
    print(f"Starting PPO training with evaluation...")
    print(f"Results will be saved to: {save_dir}")
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_length = 0
        rollout_time = time.time()
        for step in range(max_steps):
            # Select action
            action_result = agent.select_action(state)
            
            # Handle action format
            actions, log_probs, value = action_result
            if isinstance(actions, list):
                action_for_env = np.array(actions)
            else:
                action_for_env = np.array([actions])
            
            # Take action
            step_result = env.step(action_for_env)
            
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            elif len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            
            # Store experience
            agent.buffer.store(state, actions, reward, value, log_probs, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        rollout_time = time.time() - rollout_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        policy_update_time = time.time()
        # Update policy
        if agent.should_update() or (episode + 1) % update_frequency == 0:
            update_start = time.time()
            agent.update()
            update_time = time.time() - update_start
            
            if episode % save_frequency == 0 and episode > 0:
                print(f"  Update took {update_time:.2f}s")
        policy_update_time = time.time() - policy_update_time

        
        # Evaluation
        if (episode + 1) % eval_frequency == 0 or episode == num_episodes - 1:
            evaluation_time = time.time()
            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            eval_stats = evaluate_agent(
                agent, eval_env, 
                num_eval_episodes=num_eval_episodes,
                max_steps=max_steps,
                deterministic=True
            )
            
            eval_results_history.append({
                'episode': episode + 1,
                'stats': eval_stats
            })
            
            # Save intermediate results
            np.save(os.path.join(save_dir, f"training_rewards_ep_{episode + 1}.npy"), 
                   np.array(episode_rewards))
            
            # Save agent checkpoint
            save_agent(agent, os.path.join(save_dir, f"agent_ep_{episode + 1}.pth"))
            
            # Plot progress
            if len(eval_results_history) > 1:
                plot_training_progress(
                    episode_rewards, eval_results_history,
                    save_path=os.path.join(save_dir, f"progress_ep_{episode + 1}.png")
                )
            evaluation_time = time.time() - evaluation_time
            print(f"  Evaluation time: {evaluation_time:.2f}s")

        # Progress reporting
        if episode % save_frequency == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-save_frequency:])
            avg_length = np.mean(episode_lengths[-save_frequency:])
            elapsed_time = time.time() - start_time
            episodes_per_sec = episode / elapsed_time
            
            print(f"\nEpisode {episode}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Episodes/sec: {episodes_per_sec:.2f}")
            print(f"  Buffer size: {len(agent.buffer.states)}")
            print(f"  Time elapsed: {elapsed_time:.1f}s")
            print(f"  Rollout time: {rollout_time:.2f}s, Policy update time: {policy_update_time:.2f}s")
    
    # Final evaluation and save
    print(f"\n--- Final Evaluation ---")
    final_eval_stats = evaluate_agent(
        agent, eval_env,
        num_eval_episodes=num_eval_episodes * 2,  # More episodes for final eval
        max_steps=max_steps,
        deterministic=True
    )
    
    # Save final results
    final_results = {
        'training_rewards': episode_rewards,
        'eval_results_history': eval_results_history,
        'final_eval_stats': final_eval_stats,
        'training_config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'update_frequency': update_frequency,
            'eval_frequency': eval_frequency,
            'action_dims': agent.action_dims
        }
    }
    
    np.save(os.path.join(save_dir, "final_results.npy"), final_results)
    save_agent(agent, os.path.join(save_dir, "final_agent.pth"))
    
    # Final plot
    plot_training_progress(
        episode_rewards, eval_results_history,
        save_path=os.path.join(save_dir, "final_progress.png")
    )
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    return episode_rewards, eval_results_history
def test_fast_training_with_eval():
    """Quick test with evaluation"""
    print("Running fast test with evaluation...")
    
    n_cells = 3
    line = ComplexLine(
        alternate=False,
        n_assemblies=n_cells,
        n_workers=3*n_cells,
        scrap_factor=0,
        step_size=10,
        info=[],
        use_graph_as_states=True
    )
    
    env = make_stacked_vec_env(
        line=line,
        simulation_end=100,
        reward="parts",
        n_envs=1,
        n_stack=1,
        track_states=['S_component']
    )
    
    sample_state = line._graph_states
    
    if hasattr(env.action_space, 'nvec'):
        action_dims = env.action_space.nvec.tolist()
    elif hasattr(env.action_space, 'n'):
        action_dims = [env.action_space.n]
    else:
        action_dims = [4]
    
    print(f"Action dimensions: {action_dims}")
    
    agent = PPOAgent(
        sample_state=sample_state,
        action_dims=action_dims,
        lr=1e-3,
        hidden_dim=32,
        batch_size=8,
        buffer_size=64
    )
    
    # Training with evaluation
    start_time = time.time()
    rewards, eval_history = train_ppo_with_evaluation(
        env, agent,
        num_episodes=50,  # Short for testing
        max_steps=50,
        update_frequency=5,
        save_frequency=10,
        eval_frequency=20,  # Evaluate every 20 episodes
        num_eval_episodes=3,  # 3 episodes for each evaluation
        save_dir="test_ppo_results"
    )
    
    test_time = time.time() - start_time
    print(f"Test completed in {test_time:.2f}s")
    
    return rewards, eval_history

# Update your main execution
if __name__ == "__main__":
    run_test = False
    
    if run_test:
        print("Running GPU test...")
        test_rewards = test_fast_training()
    else:
        print("Running full GPU-optimized training...")
        
        n_cells = 3
        line_name = "ComplexLine"
        line = ComplexLine(
            alternate=False,
            n_assemblies=n_cells,
            n_workers=3*n_cells,
            scrap_factor=1/n_cells,
            step_size=2,
            info=[],
            use_graph_as_states=True
        )
            
        env = make_stacked_vec_env(
            line=line,
            simulation_end=4000,
            reward="parts",
            n_envs=1,
            n_stack=1,
            track_states=['S_component']
        )
        
        sample_state = line._graph_states
        
        if hasattr(env.action_space, 'nvec'):
            action_dims = env.action_space.nvec.tolist()
        elif hasattr(env.action_space, 'n'):
            action_dims = [env.action_space.n]
        else:
            action_dims = [4]
        
        print(f"Action dimensions: {action_dims}")
        
        # Create GPU-optimized agent
        agent = PPOAgent(
            sample_state=sample_state,
            action_dims=action_dims,
            lr=1e-3,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
            hidden_dim=16,  # Slightly larger for GPU efficiency
            batch_size=32,  # Larger batch size for GPU
            buffer_size=2048,  # Larger buffer
            device=DEVICE  # Explicitly set device
        )
        
        print("Starting GPU-optimized PPO training...")
        rewards, eval_history = train_ppo_with_evaluation_gpu(  # Use GPU-optimized version
            env, agent,
            num_episodes=2000,
            max_steps=1000,
            update_frequency=20,
            save_frequency=50,
            eval_frequency=200,
            num_eval_episodes=10,
            save_dir=f"ppo_gnn_results_{line_name}_gpu"
        )
        
        print("GPU training completed!")