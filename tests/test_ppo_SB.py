import wandb
import ast
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union

from lineflow.helpers import get_device
from lineflow.learning.helpers import make_stacked_vec_env
from lineflow.learning.curriculum import CurriculumLearningCallback
from lineflow.examples import (
    MultiProcess,
    WorkerAssignment,
    ComplexLine,
    WaitingTime,
    WaterLine,
)

from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, HGTConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType

from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback

## test config
test_config = {
    'env': "worker_assignment",
    'n_cells': 3,
    'model': "PPO",
    'learning_rate': 0.0003,
    'ent_coef': 0.1,
    'n_stack': 1,
    'n_steps': 10,  # Reduced for testing
    'n_envs': 1,
    'seed': 0,
    'total_steps': 100,  # Reduced for testing
    'log_dir': "./logs",
    'simulation_end': 100,  # Reduced for testing
    'gamma': 0.99,
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'normalize_advantage': False,
    'recurrent': False,
    'deterministic': False,
    'curriculum': False,
    'info': [],
    'eval_reward': "parts",
    'rollout_reward': "parts",
    'simulation_step_size': 2,
    'use_graph_as_states': True,  # Added this flag
}


# Helper functions for graph conversion
def _convert_dict_to_hetero_graph(obs_dict):
    """Convert dictionary format observation back to HeteroData object"""
    hetero_data = HeteroData()
    
    for key, value in obs_dict.items():
        if key.endswith('_x'):
            node_type = key[:-2]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            # Only add if there are actual nodes
            if value.shape[0] > 0:
                hetero_data[node_type].x = value
            
        elif key.startswith('edge_index_'):
            parts = key.split('_')[2:]
            if len(parts) >= 3:
                edge_type = (parts[0], parts[1], parts[2])
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.long)
                # Only add if there are actual edges
                if value.shape[1] > 0:  # edge_index has shape [2, num_edges]
                    hetero_data[edge_type].edge_index = value
                
        elif key.startswith('edge_attr_'):
            parts = key.split('_')[2:]
            if len(parts) >= 3:
                edge_type = (parts[0], parts[1], parts[2])
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                # Only add if there are actual edge attributes
                if value.shape[0] > 0:
                    hetero_data[edge_type].edge_attr = value
    
    return hetero_data

def _convert_hetero_graph_to_dict(hetero_data):
    """Convert HeteroData back to dictionary format compatible with gymnasium spaces.Dict"""
    obs_dict = {}
    
    # Convert node features to dictionary format
    for node_type in hetero_data.node_types:
        if node_type in hetero_data.x_dict:
            node_features = hetero_data.x_dict[node_type]
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.detach().cpu().numpy()
            
            # Ensure consistent dtype and shape
            node_features = np.asarray(node_features, dtype=np.float32)
            obs_dict[f"{node_type}_x"] = node_features
    
    # Convert edge indices and edge attributes to dictionary format  
    for edge_type in hetero_data.edge_types:
        edge_type_str = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
        
        # Handle edge indices
        if edge_type in hetero_data.edge_index_dict:
            edge_index = hetero_data.edge_index_dict[edge_type]
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.detach().cpu().numpy()
            
            # Ensure consistent dtype for edge indices
            edge_index = np.asarray(edge_index, dtype=np.int64)
            obs_dict[f"edge_index_{edge_type_str}"] = edge_index
        
        # Handle edge attributes - FIXED: Check if edge_attr actually exists
        edge_store = hetero_data[edge_type]
        if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
            edge_attr = edge_store.edge_attr
            if isinstance(edge_attr, torch.Tensor):
                edge_attr = edge_attr.detach().cpu().numpy()
            
            # Ensure consistent dtype for edge attributes
            edge_attr = np.asarray(edge_attr, dtype=np.float32)
            obs_dict[f"edge_attr_{edge_type_str}"] = edge_attr
    
    return obs_dict

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        # project input features to unified hidden dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        # Heterogeneous graph convolution layers
        # pass all the 
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class HeteroGraphFeatureExtractor_test(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: spaces.Dict,
                 data: HeteroData,
                 hidden_channels: int = 64, 
                 out_channels: int = 64, 
                 num_heads: int = 4, 
                 num_layers: int = 2, 
                 ):
        # TODO: add calculations for the features_dim before the calling the super().__init__
        # the aggregation of the feature dim used for the policy and value function can be the following possibilities
        # 1. stack of all of the features and then pass it through the mlp for the policy
        # 2. use the relevant node and edge features for the task and available action spaces
        # 3. use pooling for each of the node and edge features
        # self.features_dim = out_channels * num_heads
        self.features_dim = 64
        super().__init__(observation_space, self.features_dim)
        """
        Initialize the HeteroGraphFeatureExtractor.

        Args:
            observation_space (spaces.Dict): The observation space.
            data (HeteroData): The heterogeneous graph data.
            features_dim (int, optional): The dimension of the input features for policy and value function.
            hidden_channels (int, optional): The number of hidden channels for the GNN.
            out_channels (int, optional): The number of output channels.
            num_heads (int, optional): The number of attention heads.
            num_layers (int, optional): The number of layers.
        """
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.data = data
        self.HGT_model = HGT(hidden_channels, out_channels, num_heads, num_layers, data)
        

        # Final aggregation to fixed size then pass it through the mlp for the policy
        self.final_projection = nn.Linear(self.hidden_channels, self.features_dim)

    def _convert_flat_vector_to_hetero_graph(flat_vector, metadata):
        """
        Convert flat vector back to HeteroData object.
        
        Args:
            flat_vector: np.ndarray - Flat vector representation
            metadata: dict - Metadata from the forward conversion
        
        Returns:
            HeteroData: Reconstructed heterogeneous graph
        """
        hetero_data = HeteroData()
        
        # Reconstruct node features
        for node_type, info in metadata['node_info'].items():
            start_idx = info['start_idx']
            length = info['length']
            padded_shape = info['padded_shape']
            original_shape = info['original_shape']
            
            # Extract the flat data
            flat_data = flat_vector[start_idx:start_idx + length]
            
            # Reshape to padded shape
            reshaped_data = flat_data.reshape(padded_shape)
            
            # Remove padding to get original shape
            if padded_shape != original_shape:
                reshaped_data = reshaped_data[:original_shape[0]]
            
            # Convert to tensor and add to hetero_data
            hetero_data[node_type].x = torch.tensor(reshaped_data, dtype=torch.float32)
        
        # Reconstruct edge indices and attributes
        for edge_info_key, info in metadata['edge_info'].items():
            start_idx = info['start_idx']
            length = info['length']
            padded_shape = info['padded_shape']
            original_shape = info['original_shape']
            
            # Extract the flat data
            flat_data = flat_vector[start_idx:start_idx + length]
            
            if info['type'] == 'edge_index':
                edge_type = edge_info_key
                
                # Reshape to padded shape
                reshaped_data = flat_data.reshape(padded_shape).astype(np.int64)
                
                # Remove padding (edges with -1 indices)
                if padded_shape != original_shape:
                    # Find valid edges (not -1)
                    valid_mask = reshaped_data[0] != -1
                    reshaped_data = reshaped_data[:, valid_mask]
                
                # Convert to tensor and add to hetero_data
                hetero_data[edge_type].edge_index = torch.tensor(reshaped_data, dtype=torch.long)
                
            elif info['type'] == 'edge_attr':
                # Extract edge_type from the key (remove '_attr' suffix)
                edge_type = edge_info_key[:-5] if edge_info_key.endswith('_attr') else edge_info_key
                edge_type = eval(edge_type) if isinstance(edge_type, str) and edge_type.startswith('(') else edge_type
                
                # Reshape to padded shape
                reshaped_data = flat_data.reshape(padded_shape)
                
                # Remove padding to get original shape
                if padded_shape != original_shape:
                    reshaped_data = reshaped_data[:original_shape[0]]
                
                # Convert to tensor and add to hetero_data
                hetero_data[edge_type].edge_attr = torch.tensor(reshaped_data, dtype=torch.float32)
        
        return hetero_data

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ## convert Dict to HeteroData
        graph_data = _convert_dict_to_hetero_graph(observations)

        graph_dict = self.HGT_model(graph_data)

        feature_for_policy = graph_dict['node_features']
        # Pass through MLP
        return self.final_projection(feature_for_policy)

