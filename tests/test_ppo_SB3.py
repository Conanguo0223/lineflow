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
        
        # Handle edge attributes
        if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
            edge_attr = hetero_data[edge_type].edge_attr
            if isinstance(edge_attr, torch.Tensor):
                edge_attr = edge_attr.detach().cpu().numpy()
            
            # Ensure consistent dtype for edge attributes
            edge_attr = np.asarray(edge_attr, dtype=np.float32)
            obs_dict[f"edge_attr_{edge_type_str}"] = edge_attr
    
    return obs_dict

## Simplified Graph Feature Extractor (avoiding HGT complexity for testing)
class SimplifiedGraphFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: spaces.Dict,
                 features_dim: int = 64,
                 hidden_dim: int = 128,
                 ):
        super().__init__(observation_space, features_dim)
        
        self.hidden_dim = hidden_dim
        self.node_types = self._extract_node_types(observation_space)
        
        # Create separate MLPs for each node type
        self.node_mlps = nn.ModuleDict()
        self.node_output_dims = {}
        
        for key, space in observation_space.spaces.items():
            if key.endswith('_x'):
                node_type = key[:-2]
                input_dim = space.shape[-1]  # Feature dimension
                self.node_mlps[node_type] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                self.node_output_dims[node_type] = hidden_dim // 2
        
        # Calculate total aggregated dimension
        total_dim = sum(self.node_output_dims.values()) if self.node_output_dims else hidden_dim
        
        # Final projection
        self.final_projection = nn.Linear(64, features_dim)
    
    def _extract_node_types(self, observation_space):
        """Extract node types from observation space"""
        node_types = []
        for key in observation_space.spaces.keys():
            if key.endswith('_x'):
                node_types.append(key[:-2])
        return node_types
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Handle batch dimension
        batch_size = 1
        if isinstance(list(observations.values())[0], torch.Tensor):
            batch_size = list(observations.values())[0].shape[0]
        
        batch_features = []
        
        for i in range(batch_size):
            # Extract single observation
            if batch_size == 1:
                obs = observations
            else:
                obs = {key: value[i] for key, value in observations.items()}
            
            # Process node features
            node_embeddings = []
            
            for node_type in self.node_types:
                key = f"{node_type}_x"
                if key in obs:
                    node_features = obs[key]
                    if not isinstance(node_features, torch.Tensor):
                        node_features = torch.tensor(node_features, dtype=torch.float32)
                    
                    if node_features.shape[0] > 0:
                        # Process through MLP and aggregate
                        processed = self.node_mlps[node_type](node_features)
                        # Global mean pooling
                        aggregated = torch.mean(processed, dim=0)
                        node_embeddings.append(aggregated)
                    else:
                        # Empty node type - use zero embedding
                        zero_embedding = torch.zeros(self.node_output_dims[node_type])
                        node_embeddings.append(zero_embedding)
                else:
                    # Missing node type - use zero embedding
                    zero_embedding = torch.zeros(self.node_output_dims.get(node_type, self.hidden_dim // 2))
                    node_embeddings.append(zero_embedding)
            
            # Concatenate all node embeddings
            if node_embeddings:
                graph_embedding = torch.cat(node_embeddings, dim=0)
            else:
                graph_embedding = torch.zeros(self.hidden_dim)
            
            batch_features.append(graph_embedding)
        
        # Stack batch features
        if batch_size == 1:
            final_features = batch_features[0].unsqueeze(0)
        else:
            final_features = torch.stack(batch_features)
        
        # Final projection
        return self.final_projection(final_features)

## make the gymnasium line environment
def _make_line(name, n_cells, info, simulation_step_size=1, curriculum=False, use_graph_as_states=False):
    if name == 'worker_assignment':
        return WorkerAssignment(
            n_assemblies=n_cells,
            with_rework=False,
            step_size=simulation_step_size,
            info=info,
            use_graph_as_states=use_graph_as_states,
        )
    # ... other environments
    raise ValueError('Unknown simulation')

def test_environment_creation():
    """Test if environment can be created successfully"""
    print("Testing environment creation...")
    
    try:
        test_env = make_stacked_vec_env(
            line=_make_line(
                test_config['env'], 
                test_config['n_cells'], 
                test_config['info'], 
                curriculum=test_config['curriculum'],
                use_graph_as_states=test_config['use_graph_as_states']
            ),
            simulation_end=test_config['simulation_end'],
            reward=test_config["rollout_reward"],
            n_envs=1,
            n_stack=test_config['n_stack'],
        )
        print("✓ Environment created successfully")
        print(f"  Observation space type: {type(test_env.observation_space)}")
        print(f"  Action space: {test_env.action_space}")
        
        # Test reset and check observation structure
        obs, _ = test_env.reset()
        print(f"  Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"    {key}: {type(value)} = {value}")
        
        return test_env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_conversion_consistency(test_env):
    """Test if Dict ↔ HeteroData conversion is consistent"""
    print("\nTesting Dict ↔ HeteroData conversion consistency...")
    
    try:
        # Get original observation
        original_obs, _ = test_env.reset()
        print(f"  Original observation keys: {list(original_obs.keys())}")
        
        # Convert Dict → HeteroData
        hetero_data = _convert_dict_to_hetero_graph(original_obs)
        print(f"  HeteroData node types: {list(hetero_data.node_types)}")
        print(f"  HeteroData edge types: {list(hetero_data.edge_types)}")
        
        # Convert HeteroData → Dict
        reconstructed_obs = _convert_hetero_graph_to_dict(hetero_data)
        print(f"  Reconstructed observation keys: {list(reconstructed_obs.keys())}")
        
        # Check consistency
        conversion_success = True
        detailed_comparison = {}
        
        # Check if all original keys are preserved (allowing for filtering of empty data)
        original_non_empty_keys = set()
        for key, value in original_obs.items():
            if hasattr(value, 'shape'):
                if key.startswith('edge_index_') and len(value.shape) >= 2:
                    # Edge index: check if has edges (shape[1] > 0)
                    if value.shape[1] > 0:
                        original_non_empty_keys.add(key)
                elif value.shape[0] > 0:
                    # Node features or edge attributes: check if has data
                    original_non_empty_keys.add(key)
            else:
                original_non_empty_keys.add(key)
        
        reconstructed_keys = set(reconstructed_obs.keys())
        
        # Check key consistency
        missing_keys = original_non_empty_keys - reconstructed_keys
        extra_keys = reconstructed_keys - original_non_empty_keys
        
        if missing_keys:
            print(f"  ⚠️  Missing keys in reconstruction: {missing_keys}")
            conversion_success = False
        
        if extra_keys:
            print(f"  ⚠️  Extra keys in reconstruction: {extra_keys}")
        
        # Check value consistency for common keys
        common_keys = original_non_empty_keys & reconstructed_keys
        for key in common_keys:
            original_value = original_obs[key]
            reconstructed_value = reconstructed_obs[key]
            
            try:
                # Convert to numpy for comparison
                if isinstance(original_value, np.ndarray):
                    orig_np = original_value
                else:
                    orig_np = np.array(original_value)
                
                if isinstance(reconstructed_value, np.ndarray):
                    recon_np = reconstructed_value
                else:
                    recon_np = np.array(reconstructed_value)
                
                # Check shapes
                shape_match = orig_np.shape == recon_np.shape
                
                # Check values (with tolerance for floating point)
                if orig_np.dtype in [np.float32, np.float64]:
                    values_match = np.allclose(orig_np, recon_np, rtol=1e-5, atol=1e-8)
                else:
                    values_match = np.array_equal(orig_np, recon_np)
                
                detailed_comparison[key] = {
                    'shape_match': shape_match,
                    'values_match': values_match,
                    'original_shape': orig_np.shape,
                    'reconstructed_shape': recon_np.shape,
                    'original_dtype': orig_np.dtype,
                    'reconstructed_dtype': recon_np.dtype
                }
                
                if not shape_match or not values_match:
                    print(f"  ❌ {key}: Shape match: {shape_match}, Values match: {values_match}")
                    print(f"      Original: {orig_np.shape} {orig_np.dtype}")
                    print(f"      Reconstructed: {recon_np.shape} {recon_np.dtype}")
                    if orig_np.size <= 20:  # Only print small arrays
                        print(f"      Original values: {orig_np.flatten()}")
                        print(f"      Reconstructed values: {recon_np.flatten()}")
                    conversion_success = False
                else:
                    print(f"  ✓ {key}: Conversion consistent")
                    
            except Exception as e:
                print(f"  ❌ {key}: Comparison failed - {e}")
                conversion_success = False
        
        # Additional HeteroData integrity checks
        print(f"\n  HeteroData integrity checks:")
        print(f"    Node types with data: {len([nt for nt in hetero_data.node_types if nt in hetero_data.x_dict and hetero_data.x_dict[nt].shape[0] > 0])}")
        print(f"    Edge types with data: {len([et for et in hetero_data.edge_types if et in hetero_data.edge_index_dict and hetero_data.edge_index_dict[et].shape[1] > 0])}")
        
        # Check if HeteroData is valid for PyTorch Geometric
        try:
            # Try to access metadata (this will fail if structure is invalid)
            metadata = hetero_data.metadata()
            print(f"    ✓ Valid PyTorch Geometric metadata: {len(metadata[0])} node types, {len(metadata[1])} edge types")
        except Exception as e:
            print(f"    ❌ Invalid PyTorch Geometric structure: {e}")
            conversion_success = False
        
        if conversion_success:
            print("\n  ✓ Dict ↔ HeteroData conversion is consistent!")
        else:
            print("\n  ❌ Dict ↔ HeteroData conversion has issues!")
        
        return conversion_success, detailed_comparison
        
    except Exception as e:
        print(f"✗ Conversion consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_feature_extractor(test_env):
    """Test if feature extractor can be created and used"""
    print("\nTesting feature extractor...")
    
    # try:
    # Use simplified feature extractor for more stable testing
    feature_extractor = SimplifiedGraphFeatureExtractor(
        observation_space=test_env.observation_space,
        features_dim=64,
        hidden_dim=128
    )
    print("✓ Simplified feature extractor created successfully")
    
    # Test with sample observation
    obs, _ = test_env.reset()
    print(f"  Input observation keys: {list(obs.keys())}")
    
    # Convert to tensors with proper batch dimension
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            # Add batch dimension
            obs_tensor[key] = torch.tensor(value, dtype=torch.float32 if 'edge_index' not in key else torch.long).unsqueeze(0)
        else:
            obs_tensor[key] = torch.tensor([value], dtype=torch.float32)
    
    print("  Tensor shapes:")
    for key, tensor in obs_tensor.items():
        print(f"    {key}: {tensor.shape}")
    
    # Process through feature extractor
    features = feature_extractor(obs_tensor)
    print(f"✓ Feature extraction successful, output shape: {features.shape}")
    
    # Test conversion within feature extractor
    print("\n  Testing conversion within feature extraction pipeline...")
    for i in range(1):  # Test single observation
        obs_single = {key: value[i] if value.shape[0] > 1 else value.squeeze(0) 
                        for key, value in obs_tensor.items()}
        
        # Test conversion
        hetero_data = _convert_dict_to_hetero_graph(obs_single)
        converted_back = _convert_hetero_graph_to_dict(hetero_data)
        
        print(f"    Observation {i}: Dict→Hetero→Dict conversion successful")
    
    return feature_extractor
        
    # except Exception as e:
    #     print(f"✗ Feature extractor test failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return None

def test_ppo_model(test_env, feature_extractor):
    """Test if PPO model can be created and trained"""
    print("\nTesting PPO model...")
    
    try:
        policy_kwargs = dict(
            features_extractor_class=SimplifiedGraphFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=64,
                hidden_dim=128
            )
        )
        
        model = PPO(
            policy='MultiInputPolicy',
            env=test_env,
            n_steps=test_config["n_steps"],
            gamma=test_config['gamma'],
            learning_rate=test_config["learning_rate"],
            use_sde=False,
            normalize_advantage=test_config['normalize_advantage'],
            device=get_device(),
            verbose=1,
            policy_kwargs=policy_kwargs
        )
        print("✓ PPO model created successfully")
        
        # Test short training
        model.learn(total_timesteps=test_config["total_steps"])
        print("✓ PPO training completed successfully")
        return model
        
    except Exception as e:
        print(f"✗ PPO model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_tests():
    """Run all tests sequentially"""
    print("=" * 50)
    print("STABLE BASELINES3 + GNN INTEGRATION TESTS")
    print("=" * 50)
    
    # Test 1: Environment creation
    test_env = test_environment_creation()
    if test_env is None:
        return
    
    # Test 1.5: Conversion consistency
    conversion_success, comparison_details = test_conversion_consistency(test_env)
    if not conversion_success:
        print("⚠️  Conversion issues detected, but continuing with tests...")
    
    # Test 2: Feature extractor
    feature_extractor = test_feature_extractor(test_env)
    if feature_extractor is None:
        return
    
    # Test 3: PPO model
    model = test_ppo_model(test_env, feature_extractor)
    if model is None:
        return
    
    print("\n" + "=" * 50)
    if conversion_success:
        print("ALL TESTS PASSED! 🎉")
        print("Your Stable Baselines3 + GNN setup is working correctly!")
    else:
        print("TESTS COMPLETED WITH WARNINGS ⚠️")
        print("The setup works but has some conversion consistency issues.")
        print("Check the detailed comparison above for more information.")
    print("=" * 50)

if __name__ == '__main__':
    run_all_tests()