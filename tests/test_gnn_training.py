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
)

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear

from wandb.integration.sb3 import WandbCallback



# line = WaitingTime(use_graph_as_states=False)
line = WaitingTime(use_graph_as_states=True)
line._graph_states

env_train = make_stacked_vec_env(
        line=line,
        simulation_end=100+1,
        reward="parts",
        n_envs=1,
        n_stack=1,
    )

state, info = env_train.reset()

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, SAGEConv,TransformerConv, HGTConv



class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in state.node_types:
            in_channels = state.x_dict[node_type].shape[1]
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, state.metadata(),
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


model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=2)

# model = HGT(state.metadata())  # metadata = (node_types, edge_types)
out = model(state.x_dict, state.edge_index_dict)
