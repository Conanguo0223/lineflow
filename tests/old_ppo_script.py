import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, Linear, HGTConv
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

class HGT_policy(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, 
                 action_dims, data):  # Changed: action_dims instead of action_dim
        super().__init__()

        # Store node types and metadata
        self.node_types = list(data.node_types)
        self.hidden_channels = hidden_channels
        self.action_dims = action_dims  # List of action dimensions [4, 4, 4, ..., 200, 4]
        self.num_actions = len(action_dims)  # 14 in your case

        # Project input features to unified hidden dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        # Heterogeneous graph convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        # Shared feature extractor
        total_features = hidden_channels * len(self.node_types)
        self.shared_layers = nn.Sequential(
            nn.Linear(total_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        
        # Separate policy heads for each action component
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_channels, action_dim) 
            for action_dim in action_dims
        ])
        
        # Single value head (shared across all actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, hetero_data):
        """Forward pass with MultiDiscrete action outputs"""
        x_dict = hetero_data.x_dict
        edge_index_dict = hetero_data.edge_index_dict
        
        # Project input features to hidden dimension
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Apply HGT convolutions
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
                device = next(iter(x_dict.values())).device if x_dict else torch.device('cpu')
                zero_features = torch.zeros(1, self.hidden_channels, device=device)
                pooled_features.append(zero_features)
        
        # Concatenate all pooled node type features
        global_features = torch.cat(pooled_features, dim=1)  # Shape: [1, hidden_channels * num_node_types]
        
        # Apply shared layers
        shared_features = self.shared_layers(global_features)
        
        # Get separate action logits for each component
        action_logits = []
        for policy_head in self.policy_heads:
            logits = policy_head(shared_features)
            action_logits.append(logits)
        
        # Get value
        value = self.value_head(shared_features)
        
        return action_logits, value  # Returns list of logits, not single tensor

class PPOAgent:
    """PPO Agent for MultiDiscrete HeteroData environments"""
    
    def __init__(self, sample_state, action_dims, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, hidden_dim=64):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dims = action_dims  # [4, 4, 4, ..., 200, 4]
        self.num_actions = len(action_dims)
        
        # Initialize policy network
        self.policy = HGT_policy(
            hidden_channels=hidden_dim,
            out_channels=None,  # Not used anymore
            num_heads=4,
            num_layers=2,
            action_dims=action_dims,  # Pass the full action dimensions
            data=sample_state
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = PPOBuffer()
        
        # Loss function
        self.mse_loss = nn.MSELoss()
    
    def select_action(self, state):
        """Select action given HeteroData state"""
        with torch.no_grad():
            action_logits_list, value = self.policy(state)
            
        # Create action distributions for each component
        actions = []
        log_probs = []
        
        for i, logits in enumerate(action_logits_list):
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
        
        return actions, log_probs, value.squeeze().item()  # FIX: Squeeze value tensor
    
    def update(self, next_value=0):
        """Update policy using PPO for MultiDiscrete actions"""
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get_batch()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Convert old_log_probs and actions to proper format
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)  # Shape: [batch, num_actions]
        actions_tensor = torch.tensor(actions, dtype=torch.long)  # Shape: [batch, num_actions]
        
        # Normalize advantages
        if len(advantages) > 1:  # FIX: Check if we have multiple samples
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            for i in range(len(states)):
                # Forward pass
                action_logits_list, value = self.policy(states[i])
                
                # Compute log probabilities and entropy for each action component
                total_log_prob = 0
                total_entropy = 0
                
                for j, logits in enumerate(action_logits_list):
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(actions_tensor[i][j])
                    entropy = dist.entropy()
                    
                    total_log_prob += log_prob
                    total_entropy += entropy
                
                # Compute ratio
                ratio = torch.exp(total_log_prob - old_log_probs_tensor[i].sum())
                
                # Compute surrogate losses
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2)
                
                # Value loss - FIX: Ensure proper tensor shapes
                value_pred = value.squeeze()  # Remove extra dimensions
                value_target = returns[i]
                value_loss = self.mse_loss(value_pred, value_target)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * total_entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
    
    def compute_gae(self, rewards, values, dones, next_value, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            else:
                delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            
            gae = delta + self.gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

# Updated PPOBuffer to handle MultiDiscrete actions
class PPOBuffer:
    """Experience buffer for PPO with MultiDiscrete actions"""
    
    def __init__(self):
        self.states = []
        self.actions = []  # Will store lists of actions
        self.rewards = []
        self.values = []
        self.log_probs = []  # Will store lists of log_probs
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)  # action is a list
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)  # log_prob is a list
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

def train_ppo(env, agent, num_episodes=10000, max_steps=4000):
    """Main training loop for MultiDiscrete actions"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle tuple return from reset
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action (returns list of actions)
            actions, log_probs, value = agent.select_action(state)
            
            # Take action in environment (convert list to numpy array)
            action_array = np.array(actions)
            step_result = env.step(action_array)

            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            elif len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            # Store experience
            agent.buffer.store(state, actions, reward, value, log_probs, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update policy
        if len(agent.buffer.states) > 0:
            # Get next value for GAE computation
            if not done:
                _, _, next_value = agent.select_action(next_state)
            else:
                next_value = 0
            
            agent.update(next_value)
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    return np.array(episode_rewards)

if __name__ == "__main__":
    # Initialize environment and agent
    n_cells = 3
    line_name = "ComplexLine"
    line = ComplexLine(
        alternate=False,
            n_assemblies=n_cells,
            n_workers=3*n_cells,
            scrap_factor=1/n_cells,
            step_size=10,
            info=[],
            use_graph_as_states=True
            )
    # line_name = "WaitingTime"
    # line = WaitingTime(
    #         info=[],
    #         processing_time_source=5,
    #         use_graph_as_states=True
    #     )
    env = make_stacked_vec_env(
        line=line,
        simulation_end=4000,
        reward="parts",
        n_envs=1,
        n_stack=1,
        track_states=['S_component']
    )
    
    sample_state = line._graph_states
    
    # Get the actual action dimensions from environment
    if hasattr(env.action_space, 'nvec'):
        action_dims = env.action_space.nvec.tolist()
    else:
        print("I have weird action dims")
        action_dims = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 200, 4]  # Your specified dims

    print(f"Action dimensions: {action_dims}")
    
    agent = PPOAgent(
        sample_state=sample_state,
        action_dims=action_dims,  # Pass the full action dimensions
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        hidden_dim=64
    )
    
    # Train the agent
    print("Starting PPO training with MultiDiscrete HeteroData states...")
    rewards = train_ppo(env, agent, num_episodes=10000)
    np.save("ppo_gnn_rewards_"+line_name+".npy", rewards)
    print("Training completed!")