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


