import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, SAGEConv,TransformerConv, HGTConv, HANConv


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
            conv = HANConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
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


# model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=3)

# # model = HGT(state.metadata())  # metadata = (node_types, edge_types)
# out = model(data.x_dict, data.edge_index_dict)

dataset = torch.load('data/complex_line_graph_n_assemblies5_waiting_time5.pt', weights_only=False)

import torch.nn as nn
import random
import numpy as np

# Training parameters
num_epochs = 1000
learning_rate = 1e-3
batch_size = 16
sequence_length = 5
temporal_model = GraphStatePredictor(hidden_channels=64, out_channels=4, num_heads=2, num_layers=3, context_window=5, data=dataset['graph'][0])

# Initialize optimizer and loss function
predictor_optimizer = torch.optim.Adam(temporal_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temporal_model = temporal_model.to(device)

# Training loop
temporal_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    max_loss = 0.0
    max_loss_timestep = -1
    # Create training sequences from all_data_sets
    for i in range(len(dataset['graph']) - sequence_length):
        # Sample from random index in the current dataset
        if len(dataset['graph']) > sequence_length:
            random_index = np.random.randint(0, len(dataset['graph']) - sequence_length)
            # print(random_index)
        else:
            random_index = 0
        
        # Use the random index for both input sequence and target
        input_sequence = dataset['graph'][random_index:random_index+sequence_length]
        target_graph = dataset['graph'][random_index+sequence_length]
        # # Get input sequence and target
        # input_sequence = dataset['graph'][i:i+sequence_length]
        # target_graph = dataset['graph'][i+sequence_length]
        # Move data to device
        input_sequence = [graph.to(device) for graph in input_sequence]
        target_dict = {node_type: features.to(device) for node_type, features in target_graph.x_dict.items()}
        
        # Forward pass
        predictor_optimizer.zero_grad()
        predictions = temporal_model(input_sequence)
        
        # Calculate loss for each node type
        total_loss = 0.0
        for node_type in predictions.keys():
            loss = criterion(predictions[node_type], target_dict[node_type])
            total_loss += loss
            max_loss = max(max_loss, loss.item())
            max_loss_timestep = random_index

        # Backward pass
        total_loss.backward()
        predictor_optimizer.step()
        
        epoch_loss += total_loss.item()
        num_batches += 1
        
        # Break if we have enough training samples
        if num_batches >= 70:  # Limit training samples per epoch
            break
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.4f}, Max Loss: {max_loss:.4f}, Max Loss Timestep: {max_loss_timestep}")
    
        # print(f"index used: {index}, results of this index: {dataset['Reward']}")
torch.save(temporal_model.state_dict(), 'temporal_model.pth')
print("Training completed!")