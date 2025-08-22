import wandb
import ast
import argparse
import os
from lineflow.helpers import get_device
from lineflow.learning.helpers import (
    make_stacked_vec_env,
)
from lineflow.learning.curriculum import CurriculumLearningCallback
from lineflow.examples import (
    WaitingTime,
    ComplexLine,
    WaterLine
)
import torch
import numpy as np
from torch_geometric.data import HeteroData
from wandb.integration.sb3 import WandbCallback
import networkx as nx
import matplotlib.pyplot as plt
import stable_baselines3

def _convert_hetero_graph_to_dict(hetero_data):
    """
    Convert HeteroData back to dictionary format for observation space compatibility
    
    Args:
        hetero_data: HeteroData object
        
    Returns:
        dict: Dictionary with node and edge information including edge attributes
    """
    obs_dict = {}
    
    # Convert node features to dictionary format
    for node_type in hetero_data.node_types:
        if node_type in hetero_data.x_dict:
            node_features = hetero_data.x_dict[node_type]
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.detach().cpu().numpy()
            obs_dict[f"{node_type}_x"] = node_features.astype(np.float32)
    
    # Convert edge indices and edge attributes to dictionary format  
    for edge_type in hetero_data.edge_types:
        edge_type_str = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
        
        # Handle edge indices
        if edge_type in hetero_data.edge_index_dict:
            edge_index = hetero_data.edge_index_dict[edge_type]
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.detach().cpu().numpy()
            obs_dict[f"edge_index_{edge_type_str}"] = edge_index.astype(np.int64)
        
        # Handle edge attributes
        if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
            edge_attr = hetero_data[edge_type].edge_attr
            if isinstance(edge_attr, torch.Tensor):
                edge_attr = edge_attr.detach().cpu().numpy()
            obs_dict[f"edge_attr_{edge_type_str}"] = edge_attr.astype(np.float32)
    
    return obs_dict

def _convert_dict_to_hetero_graph(obs_dict):
    """
    Convert dictionary format observation back to HeteroData object
    
    Args:
        obs_dict: Dictionary with node and edge information in the format:
                 - "{node_type}_x": node features (numpy array)
                 - "edge_index_{src}_{rel}_{dst}": edge indices (numpy array)
                 - "edge_attr_{src}_{rel}_{dst}": edge attributes (numpy array)
        
    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph
    """
    hetero_data = HeteroData()
    
    # Extract node features, edge indices, and edge attributes
    node_features = {}
    edge_indices = {}
    edge_attributes = {}
    
    for key, value in obs_dict.items():
        if key.endswith('_x'):
            # This is node feature data
            node_type = key[:-2]  # Remove '_x' suffix
            
            # Convert to tensor if needed
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            
            node_features[node_type] = value
            
        elif key.startswith('edge_index_'):
            # This is edge index data
            # Parse edge type from key: "edge_index_{src}_{rel}_{dst}"
            parts = key.split('_')[2:]  # Remove 'edge_index_' prefix
            
            if len(parts) >= 3:
                src_type = parts[0]
                rel_type = parts[1] 
                dst_type = parts[2]
                edge_type = (src_type, rel_type, dst_type)
            else:
                # Fallback for simpler edge naming
                edge_type = tuple(parts)
            
            # Convert to tensor if needed
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.long)
            
            edge_indices[edge_type] = value
            
        elif key.startswith('edge_attr_'):
            # This is edge attribute data
            # Parse edge type from key: "edge_attr_{src}_{rel}_{dst}"
            parts = key.split('_')[2:]  # Remove 'edge_attr_' prefix
            
            if len(parts) >= 3:
                src_type = parts[0]
                rel_type = parts[1] 
                dst_type = parts[2]
                edge_type = (src_type, rel_type, dst_type)
            else:
                # Fallback for simpler edge naming
                edge_type = tuple(parts)
            
            # Convert to tensor if needed
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            
            edge_attributes[edge_type] = value
    
    # Add node features to HeteroData
    for node_type, features in node_features.items():
        hetero_data[node_type].x = features
    
    # Add edge indices to HeteroData
    for edge_type, edge_index in edge_indices.items():
        hetero_data[edge_type].edge_index = edge_index
    
    # Add edge attributes to HeteroData
    for edge_type, edge_attr in edge_attributes.items():
        hetero_data[edge_type].edge_attr = edge_attr
    
    return hetero_data

def test_hetero_data_equality(hetero_data1, hetero_data2, tolerance=1e-6, verbose=True):
    """
    Test if two HeteroData objects are equivalent
    
    Args:
        hetero_data1: First HeteroData object
        hetero_data2: Second HeteroData object
        tolerance: Numerical tolerance for floating point comparisons
        verbose: Whether to print detailed comparison results
        
    Returns:
        bool: True if the HeteroData objects are equivalent
    """
    if verbose:
        print("Testing HeteroData equality...")
    
    success = True
    
    # 1. Check if both are HeteroData objects
    if not isinstance(hetero_data1, HeteroData) or not isinstance(hetero_data2, HeteroData):
        if verbose:
            print("❌ One or both objects are not HeteroData instances")
        return False
    
    # 2. Check node types
    node_types1 = set(hetero_data1.node_types)
    node_types2 = set(hetero_data2.node_types)
    
    if node_types1 != node_types2:
        if verbose:
            print(f"❌ Node types don't match!")
            print(f"   Data1: {sorted(node_types1)}")
            print(f"   Data2: {sorted(node_types2)}")
        success = False
    elif verbose:
        print(f"✅ Node types match: {sorted(node_types1)}")
    
    # 3. Check edge types
    edge_types1 = set(hetero_data1.edge_types)
    edge_types2 = set(hetero_data2.edge_types)
    
    if edge_types1 != edge_types2:
        if verbose:
            print(f"❌ Edge types don't match!")
            print(f"   Data1: {sorted(edge_types1)}")
            print(f"   Data2: {sorted(edge_types2)}")
        success = False
    elif verbose:
        print(f"✅ Edge types match: {sorted(edge_types1)}")
    
    # 4. Check node features
    for node_type in node_types1.intersection(node_types2):
        # Check if both have node features
        has_x1 = hasattr(hetero_data1[node_type], 'x') and hetero_data1[node_type].x is not None
        has_x2 = hasattr(hetero_data2[node_type], 'x') and hetero_data2[node_type].x is not None
        
        if has_x1 != has_x2:
            if verbose:
                print(f"❌ Node {node_type}: x feature presence mismatch")
            success = False
            continue
            
        if has_x1 and has_x2:
            x1 = hetero_data1[node_type].x
            x2 = hetero_data2[node_type].x
            
            # Check shapes
            if x1.shape != x2.shape:
                if verbose:
                    print(f"❌ Node {node_type}: x shape mismatch ({x1.shape} vs {x2.shape})")
                success = False
                continue
            
            # Check values
            if not torch.allclose(x1, x2, atol=tolerance, rtol=tolerance):
                diff = torch.abs(x1 - x2).max().item()
                if verbose:
                    print(f"❌ Node {node_type}: x values differ by max {diff:.2e}")
                success = False
            elif verbose:
                print(f"✅ Node {node_type}: x features match")
    
    # 5. Check edge indices and attributes
    for edge_type in edge_types1.intersection(edge_types2):
        # Check edge indices
        has_edge_index1 = hasattr(hetero_data1[edge_type], 'edge_index') and hetero_data1[edge_type].edge_index is not None
        has_edge_index2 = hasattr(hetero_data2[edge_type], 'edge_index') and hetero_data2[edge_type].edge_index is not None
        
        if has_edge_index1 != has_edge_index2:
            if verbose:
                print(f"❌ Edge {edge_type}: edge_index presence mismatch")
            success = False
            continue
            
        if has_edge_index1 and has_edge_index2:
            edge_index1 = hetero_data1[edge_type].edge_index
            edge_index2 = hetero_data2[edge_type].edge_index
            
            # Check shapes
            if edge_index1.shape != edge_index2.shape:
                if verbose:
                    print(f"❌ Edge {edge_type}: edge_index shape mismatch ({edge_index1.shape} vs {edge_index2.shape})")
                success = False
                continue
            
            # Check values (exact match for indices)
            if not torch.equal(edge_index1, edge_index2):
                if verbose:
                    print(f"❌ Edge {edge_type}: edge_index values don't match")
                success = False
            elif verbose:
                print(f"✅ Edge {edge_type}: edge_index matches")
        
        # Check edge attributes
        has_edge_attr1 = hasattr(hetero_data1[edge_type], 'edge_attr') and hetero_data1[edge_type].edge_attr is not None
        has_edge_attr2 = hasattr(hetero_data2[edge_type], 'edge_attr') and hetero_data2[edge_type].edge_attr is not None
        
        if has_edge_attr1 != has_edge_attr2:
            if verbose:
                print(f"❌ Edge {edge_type}: edge_attr presence mismatch")
            success = False
            continue
            
        if has_edge_attr1 and has_edge_attr2:
            edge_attr1 = hetero_data1[edge_type].edge_attr
            edge_attr2 = hetero_data2[edge_type].edge_attr
            
            # Check shapes
            if edge_attr1.shape != edge_attr2.shape:
                if verbose:
                    print(f"❌ Edge {edge_type}: edge_attr shape mismatch ({edge_attr1.shape} vs {edge_attr2.shape})")
                success = False
                continue
            
            # Check values
            if not torch.allclose(edge_attr1, edge_attr2, atol=tolerance, rtol=tolerance):
                diff = torch.abs(edge_attr1 - edge_attr2).max().item()
                if verbose:
                    print(f"❌ Edge {edge_type}: edge_attr values differ by max {diff:.2e}")
                success = False
            elif verbose:
                print(f"✅ Edge {edge_type}: edge_attr matches")
    
    # 6. Check for any additional attributes
    all_attrs1 = set()
    all_attrs2 = set()
    
    # Collect all node and edge store keys
    for node_type in hetero_data1.node_types:
        all_attrs1.update(hetero_data1[node_type].keys())
    for edge_type in hetero_data1.edge_types:
        all_attrs1.update(hetero_data1[edge_type].keys())
        
    for node_type in hetero_data2.node_types:
        all_attrs2.update(hetero_data2[node_type].keys())
    for edge_type in hetero_data2.edge_types:
        all_attrs2.update(hetero_data2[edge_type].keys())
    
    # Remove standard attributes we already checked
    standard_attrs = {'x', 'edge_index', 'edge_attr'}
    extra_attrs1 = all_attrs1 - standard_attrs
    extra_attrs2 = all_attrs2 - standard_attrs
    
    if extra_attrs1 != extra_attrs2:
        if verbose:
            print(f"❌ Additional attributes don't match!")
            print(f"   Data1 extras: {sorted(extra_attrs1)}")
            print(f"   Data2 extras: {sorted(extra_attrs2)}")
        success = False
    elif extra_attrs1 and verbose:
        print(f"✅ Additional attributes match: {sorted(extra_attrs1)}")
    
    # Final result
    if success:
        if verbose:
            print("🎉 HeteroData objects are equivalent!")
        return True
    else:
        if verbose:
            print("❌ HeteroData objects are NOT equivalent!")
        return False


def test_roundtrip_conversion(original_hetero_data, tolerance=1e-6):
    """
    Test roundtrip conversion: HeteroData -> dict -> HeteroData
    
    Args:
        original_hetero_data: Original HeteroData object
        tolerance: Numerical tolerance for comparisons
        
    Returns:
        bool: True if roundtrip conversion preserves the data
    """
    print("Testing roundtrip conversion...")
    
    try:
        # Convert to dict
        obs_dict = _convert_hetero_graph_to_dict(original_hetero_data)
        print(f"✅ Successfully converted to dict with {len(obs_dict)} keys")
        
        # Convert back to HeteroData
        reconstructed = _convert_dict_to_hetero_graph(obs_dict)
        print(f"✅ Successfully reconstructed HeteroData")
        
        # Test equality
        is_equal = test_hetero_data_equality(
            original_hetero_data, 
            reconstructed, 
            tolerance=tolerance,
            verbose=True
        )
        
        return is_equal
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False


# Quick equality check function for simple cases
def hetero_data_equal(data1, data2, tolerance=1e-6):
    """
    Simple equality check without verbose output
    
    Args:
        data1, data2: HeteroData objects to compare
        tolerance: Numerical tolerance
        
    Returns:
        bool: True if equal
    """
    return test_hetero_data_equality(data1, data2, tolerance=tolerance, verbose=False)
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, SAGEConv,TransformerConv, HGTConv

class GraphStatePredictor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, context_window=5):
        super().__init__()
        self.context_window = context_window
        self.hidden_channels = hidden_channels
        self.data = data
        # Linear layers for each node type to project to hidden dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)
        
        # HGT layers for processing individual graphs
        self.graph_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.graph_convs.append(conv)
        
        # Temporal transformer to process sequence of graph states
        self.temporal_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=num_heads,
                dim_feedforward=hidden_channels * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection for each node type
        self.output_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            out_features = data.x_dict[node_type].shape[1]
            self.output_dict[node_type] = Linear(hidden_channels, out_features)
    
    def encode_single_graph(self, x_dict, edge_index_dict):
        # Project node features to hidden dimension
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        
        # Apply HGT convolutions
        for conv in self.graph_convs:
            x_dict = conv(x_dict, edge_index_dict)
        
        return x_dict
    
    def forward(self, graph_sequence):
        """
        Args:
            graph_sequence: List of HeteroData objects representing graph states
        Returns:
            Predicted next graph state as dict of node type tensors
        """
        batch_size = len(graph_sequence)
        
        # Encode each graph in the sequence
        encoded_sequence = {}
        for node_type in self.data.node_types:
            encoded_sequence[node_type] = []
        
        for graph in graph_sequence:
            encoded_graph = self.encode_single_graph(graph.x_dict, graph.edge_index_dict)
            for node_type in self.data.node_types:
                encoded_sequence[node_type].append(encoded_graph[node_type])
        
        # Apply temporal transformer for each node type
        predictions = {}
        for node_type in self.data.node_types:
            # Stack temporal sequence: [seq_len, num_nodes, hidden_dim]
            node_sequence = torch.stack(encoded_sequence[node_type], dim=0)
            num_nodes = node_sequence.shape[1]
            
            # Reshape for transformer: [num_nodes, seq_len, hidden_dim]
            node_sequence = node_sequence.transpose(0, 1)
            
            # Apply temporal transformer to each node independently
            temporal_output = []
            for node_idx in range(num_nodes):
                node_temporal = node_sequence[node_idx].unsqueeze(0)  # [1, seq_len, hidden_dim]
                transformed = self.temporal_transformer(node_temporal)
                temporal_output.append(transformed[:, -1, :])  # Take last timestep
            
            temporal_features = torch.cat(temporal_output, dim=0)  # [num_nodes, hidden_dim]
            
            # Project to output dimension
            predictions[node_type] = self.output_dict[node_type](temporal_features)
        
        return predictions


class GraphStatePredictor_no_temporal(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.data = data
        # Linear layers for each node type to project to hidden dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)
            
        # HGT layers for processing individual graphs
        self.graph_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.graph_convs.append(conv)
        
        # Temporal transformer to process sequence of graph states
        self.temporal_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=num_heads,
                dim_feedforward=hidden_channels * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection for each node type
        self.output_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            out_features = data.x_dict[node_type].shape[1]
            self.output_dict[node_type] = Linear(hidden_channels, out_features)
    
    def encode_single_graph(self, x_dict, edge_index_dict):
        # Project node features to hidden dimension
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        
        # Apply HGT convolutions
        for conv in self.graph_convs:
            x_dict = conv(x_dict, edge_index_dict)
        
        return x_dict
    
    def forward(self, graph_sequence):
        """
        Args:
            graph_sequence: List of HeteroData objects representing graph states
        Returns:
            Predicted next graph state as dict of node type tensors
        """
        batch_size = len(graph_sequence)
        
        # Encode each graph in the sequence
        encoded_sequence = {}
        for node_type in self.data.node_types:
            encoded_sequence[node_type] = []
        
        for graph in graph_sequence:
            encoded_graph = self.encode_single_graph(graph.x_dict, graph.edge_index_dict)
            for node_type in self.data.node_types:
                encoded_sequence[node_type].append(encoded_graph[node_type])
        
        # Apply temporal transformer for each node type
        predictions = {}
        for node_type in self.data.node_types:
            # Stack temporal sequence: [seq_len, num_nodes, hidden_dim]
            node_sequence = torch.stack(encoded_sequence[node_type], dim=0)
            num_nodes = node_sequence.shape[1]
            
            # Reshape for transformer: [num_nodes, seq_len, hidden_dim]
            node_sequence = node_sequence.transpose(0, 1)
            
            # Apply temporal transformer to each node independently
            temporal_output = []
            for node_idx in range(num_nodes):
                node_temporal = node_sequence[node_idx].unsqueeze(0)  # [1, seq_len, hidden_dim]
                transformed = self.temporal_transformer(node_temporal)
                temporal_output.append(transformed[:, -1, :])  # Take last timestep
            
            temporal_features = torch.cat(temporal_output, dim=0)  # [num_nodes, hidden_dim]
            
            # Project to output dimension
            predictions[node_type] = self.output_dict[node_type](temporal_features)
        
        return predictions

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_channels = data.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

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
