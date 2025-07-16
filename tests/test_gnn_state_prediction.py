import numpy as np
from lineflow.simulation import (
    Source,
    Sink,
    Line,
    Assembly,
)
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import HGTConv
from typing import Dict


def make_agent_fixed_waiting_time(line, waiting_time):
    waiting_times = line['S_component'].state['waiting_time'].categories

    def agent(state, env):
        """
        A policy that can effectively set float waiting times by
        alternating between ints
        """

        index = np.argmin(np.abs(waiting_times - waiting_time))
        actions = {}
        actions['S_component'] = {'waiting_time': index}
        return actions
    return agent


def compute_optimal_waiting_time(line):
    time_assembly = line['Assembly'].processing_time*1.1 + 1 + 1 + 1.1
    time_source = line['S_component'].processing_time*1.1 + 1.1
    return time_assembly-time_source


def make_optimal_agent(line):

    waiting_times = line['S_component'].state['waiting_time'].categories
    processing_time_source = line['S_component'].processing_time

    def agent(state, env):
        """
        A policy that can effectively set float waiting times by
        alternating between ints
        """
        time_assembly = state['Assembly']['processing_time'].value + 1 + 1 + 1.1
        time_source = processing_time_source*1.1 + 1.1
        waiting_time = time_assembly - time_source

        index = np.argmin(np.abs(waiting_times - waiting_time))
        actions = {}
        actions['S_component'] = {'waiting_time': index}
        return actions
    return agent


class WTAssembly(Assembly):

    def __init__(
        self,
        name,
        R=0.75,
        t_jump_max=2000,
        **kwargs,
    ):

        self.R = R
        self.t_jump_max = t_jump_max
        self.trigger_time = None
        self.factor = None

        super().__init__(name=name,  **kwargs)

    def init(self, random):
        """
        Function that is called after line is built, so all available information is present
        """
        super().init(random)

        self._sample_trigger_time()

    def _compute_scaling_factor(self, T_jump, E=3.1):

        T = self.processing_time
        S = self.processing_std
        T_sim = self.t_jump_max*2

        return 1/T*((T_jump*(T+S+E)) / ((self.R-1)*T_sim+T_jump) - S -E)


    def _sample_trigger_time(self):

        self.t_jump = np.random.uniform(
            0.8*self.t_jump_max,
            self.t_jump_max,
        )

        self.factor = self._compute_scaling_factor(self.t_jump)
        self.trigger_time = self.random.uniform(0.25, 0.75)*self.t_jump_max

    def _sample_exp_time(self, time=None, scale=None, rework_probability=0):
        """
        Samples a time from an exponential distribution
        """
        coeff = self.get_performance_coefficient()
        if self.trigger_time < self.env.now < self.trigger_time + self.t_jump:
            factor = self.factor
        else: 
            factor = 1

        return time*factor*coeff + self.random.exponential(scale=scale)



class WaitingTime(Line):
    def __init__(
        self, 
        processing_time_source=5, 
        transition_time=5, 
        with_jump=False,
        t_jump_max=None,
        assembly_condition=35,
        scrap_factor=1,
        R=0.75,
        **kwargs,
    ):
        self.processing_time_source = processing_time_source
        self.transition_time = transition_time
        self.with_jump = with_jump
        self.t_jump_max = t_jump_max
        self.assembly_condition = assembly_condition
        self.R = R

        if self.with_jump:
            assert self.t_jump_max is not None
        super().__init__(scrap_factor=scrap_factor, **kwargs)

    def build(self):

        source_main = Source(
            'S_main',
            position=(300, 300),
            processing_time=0,
            carrier_capacity=2,
            actionable_waiting_time=False,
            unlimited_carriers=True,
        )

        source_component = Source(
            'S_component',
            position=(500, 450),
            processing_time=self.processing_time_source,
            waiting_time=0,
            waiting_time_step=1,
            carrier_capacity=1,
            carrier_specs={
                'carrier': {"Part": {"Assembly": {"assembly_condition": self.assembly_condition}}}
            },
            unlimited_carriers=True,
            actionable_waiting_time=True,
        )

        if self.with_jump:
            assembly = WTAssembly(
                'Assembly',
                t_jump_max=self.t_jump_max,
                position=(500, 300),
                R=self.R,
                processing_time=20,
                NOK_part_error_time=5,
            )
        else:
            assembly = Assembly(
                'Assembly',
                position=(500, 300),
                processing_time=20,
                NOK_part_error_time=5,
            )

        sink = Sink('Sink', processing_time=0, position=(700, 300))

        assembly.connect_to_component_input(
            station=source_component,
            capacity=3,
            transition_time=self.transition_time,
        )
        assembly.connect_to_input(source_main, capacity=2, transition_time=2)
        sink.connect_to_input(assembly, capacity=2, transition_time=2)



class ImprovedHGT(torch.nn.Module):
    """Enhanced HGT with better initialization and regularization"""
    
    def __init__(self, metadata, node_feature_dims: Dict[str, int]):
        super().__init__()
        self.hidden_channels: int = 64
        self.out_channels: int = 4
        self.num_heads: int = 2
        self.num_layers: int = 1
        self.dropout: float = 0.1
        self.metadata = metadata
        
        # Input projection layers with proper initialization
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:  # node_types
            self.lin_dict[node_type] = nn.Sequential(
                Linear(node_feature_dims[node_type], self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        
        # HGT layers with residual connections
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            conv = HGTConv(
                self.hidden_channels, 
                self.hidden_channels, 
                metadata,
                self.num_heads
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_channels))
        
        # Output projection
        self.output_proj = nn.Sequential(
            Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Linear(self.hidden_channels, self.out_channels)
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

if __name__ == '__main__':
    line = WaitingTime(use_graph_as_states=True, step_size=100)
    agent = make_optimal_agent(line)
    collected_data = line.run(simulation_end=4000, agent=agent, visualize=False, collect_data=True)
    print(line.get_n_parts_produced())


# import numpy as np
# import torch
# from torch import nn
# from torch.nn import Linear
# from torch_geometric.nn import HGTConv
# from torch_geometric.data import HeteroData
# from typing import Dict, List, Tuple
# import matplotlib.pyplot as plt
# from lineflow.simulation import (
#     Source,
#     Sink,
#     Line,
#     Assembly,
# )

# # ...existing code (WTAssembly, WaitingTime classes)...

# class GraphStateCollector:
#     """Collects graph states and transitions for training"""
    
#     def __init__(self, line):
#         self.line = line
#         self.states = []
#         self.next_states = []
#         self.actions = []
#         self.timesteps = []
        
#     def collect_transition(self, current_state: HeteroData, action: dict, next_state: HeteroData, timestep: int):
#         """Collect a single transition"""
#         self.states.append(self._clone_hetero_data(current_state))
#         self.next_states.append(self._clone_hetero_data(next_state))
#         self.actions.append(action.copy())
#         self.timesteps.append(timestep)
    
#     def _clone_hetero_data(self, data: HeteroData) -> HeteroData:
#         """Deep copy HeteroData"""
#         new_data = HeteroData()
#         for key, value in data.items():
#             new_data[key] = value.clone() if hasattr(value, 'clone') else value
#         return new_data
    
#     def get_dataset(self) -> Tuple[List[HeteroData], List[HeteroData], List[dict]]:
#         """Return collected dataset"""
#         return self.states, self.next_states, self.actions

# class StatePredictor(torch.nn.Module):
#     """GNN model to predict next states"""
    
#     def __init__(self, metadata, node_feature_dims: Dict[str, int]):
#         super().__init__()
#         self.hidden_channels = 64
#         self.num_heads = 2
#         self.num_layers = 2
#         self.dropout = 0.1
#         self.metadata = metadata
        
#         # Input projection layers
#         self.lin_dict = torch.nn.ModuleDict()
#         for node_type in metadata[0]:  # node_types
#             self.lin_dict[node_type] = nn.Sequential(
#                 Linear(node_feature_dims[node_type], self.hidden_channels),
#                 nn.LayerNorm(self.hidden_channels),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout)
#             )
        
#         # HGT encoder layers
#         self.convs = torch.nn.ModuleList()
#         self.norms = torch.nn.ModuleList()
#         for _ in range(self.num_layers):
#             conv = HGTConv(
#                 self.hidden_channels, 
#                 self.hidden_channels, 
#                 metadata,
#                 self.num_heads
#             )
#             self.convs.append(conv)
#             self.norms.append(nn.LayerNorm(self.hidden_channels))
        
#         # State prediction heads for each node type
#         self.prediction_heads = torch.nn.ModuleDict()
#         for node_type in metadata[0]:
#             self.prediction_heads[node_type] = nn.Sequential(
#                 Linear(self.hidden_channels, self.hidden_channels),
#                 nn.ReLU(),
#                 nn.Dropout(self.dropout),
#                 Linear(self.hidden_channels, node_feature_dims[node_type])
#             )
        
#         self._init_weights()
    
#     def _init_weights(self):
#         """Proper weight initialization"""
#         for m in self.modules():
#             if isinstance(m, Linear):
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias)
    
#     def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         # Input projection
#         x_dict = {
#             node_type: self.lin_dict[node_type](x) 
#             for node_type, x in x_dict.items()
#         }
        
#         # HGT encoder layers with residual connections
#         for conv, norm in zip(self.convs, self.norms):
#             x_dict_new = conv(x_dict, edge_index_dict)
#             # Apply residual connection and normalization
#             x_dict = {
#                 node_type: norm(x_dict_new[node_type] + x_dict[node_type])
#                 for node_type in x_dict.keys()
#             }
        
#         # State prediction
#         predictions = {}
#         for node_type, features in x_dict.items():
#             predictions[node_type] = self.prediction_heads[node_type](features)
        
#         return predictions

# def collect_training_data(line, num_episodes=10, steps_per_episode=100):
#     """Collect training data by running simulation with various agents"""
    
#     collector = GraphStateCollector(line)
    
#     for episode in range(num_episodes):
#         print(f"Collecting episode {episode + 1}/{num_episodes}")
        
#         # Reset line for new episode
#         line.reset()
        
#         # Use random agent for data diversity
#         def random_agent(state, env):
#             waiting_times = line['S_component'].state['waiting_time'].categories
#             actions = {}
#             actions['S_component'] = {'waiting_time': np.random.choice(len(waiting_times))}
#             return actions
        
#         # Run simulation and collect states
#         current_state = None
#         timestep = 0
#         terminated = False
#         while timestep < steps_per_episode and not terminated:
#             # Get current graph state
#             if hasattr(line, '_graph_states'):
#                 current_graph_state = line._graph_states
#             else:
#                 # If graph states not available, skip
#                 break
            
#             # Get action from agent
#             line_state = line.state
#             action = random_agent(line_state, line.env)
            
#             # Store current state
#             if current_state is not None:
#                 collector.collect_transition(
#                     current_state, 
#                     previous_action, 
#                     current_graph_state, 
#                     timestep
#                 )
            
#             # Apply action and step
#             line.step(action)
            
#             # Update for next iteration
#             current_state = current_graph_state
#             previous_action = action
#             timestep += 1
    
#     return collector

# def train_state_predictor(collector: GraphStateCollector, num_epochs=100):
#     """Train the state prediction model"""
    
#     states, next_states, actions = collector.get_dataset()
    
#     if len(states) == 0:
#         print("No training data collected!")
#         return None
    
#     # Get metadata from first state
#     sample_state = states[0]
#     metadata = sample_state.metadata()
    
#     # Calculate node feature dimensions
#     node_feature_dims = {}
#     for node_type in metadata[0]:
#         if node_type in sample_state.x_dict:
#             node_feature_dims[node_type] = sample_state.x_dict[node_type].shape[1]
    
#     # Create model
#     model = StatePredictor(metadata, node_feature_dims)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
    
#     # Training loop
#     losses = []
    
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         num_batches = 0
        
#         # Simple batch processing (can be improved with DataLoader)
#         for i in range(len(states)):
#             current_state = states[i]
#             target_state = next_states[i]
            
#             # Forward pass
#             optimizer.zero_grad()
#             predictions = model(current_state.x_dict, current_state.edge_index_dict)
            
#             # Calculate loss for each node type
#             total_loss = 0
#             for node_type in predictions.keys():
#                 if node_type in target_state.x_dict:
#                     pred = predictions[node_type]
#                     target = target_state.x_dict[node_type]
#                     loss = criterion(pred, target)
#                     total_loss += loss
            
#             # Backward pass
#             if total_loss > 0:
#                 total_loss.backward()
#                 optimizer.step()
#                 epoch_loss += total_loss.item()
#                 num_batches += 1
        
#         avg_loss = epoch_loss / max(num_batches, 1)
#         losses.append(avg_loss)
        
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
    
#     return model, losses

# def evaluate_model(model, collector: GraphStateCollector):
#     """Evaluate the trained model"""
    
#     states, next_states, _ = collector.get_dataset()
    
#     if len(states) == 0:
#         print("No data to evaluate!")
#         return
    
#     model.eval()
#     total_error = 0.0
#     num_predictions = 0
    
#     with torch.no_grad():
#         for i in range(len(states)):
#             current_state = states[i]
#             target_state = next_states[i]
            
#             predictions = model(current_state.x_dict, current_state.edge_index_dict)
            
#             # Calculate mean absolute error
#             for node_type in predictions.keys():
#                 if node_type in target_state.x_dict:
#                     pred = predictions[node_type]
#                     target = target_state.x_dict[node_type]
#                     error = torch.mean(torch.abs(pred - target))
#                     total_error += error.item()
#                     num_predictions += 1
    
#     avg_error = total_error / max(num_predictions, 1)
#     print(f"Average Prediction Error: {avg_error:.6f}")
    
#     return avg_error

# def plot_training_curves(losses):
#     """Plot training loss curves"""
#     plt.figure(figsize=(10, 6))
#     plt.plot(losses)
#     plt.title('Training Loss Over Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.show()

# if __name__ == '__main__':
#     # Create line with graph states enabled
#     line = WaitingTime(use_graph_as_states=True)
    
#     print("Collecting training data...")
#     collector = collect_training_data(line, num_episodes=5, steps_per_episode=50)
    
#     print(f"Collected {len(collector.states)} state transitions")
    
#     if len(collector.states) > 0:
#         print("Training state predictor...")
#         model, losses = train_state_predictor(collector, num_epochs=50)
        
#         if model is not None:
#             print("Evaluating model...")
#             evaluate_model(model, collector)
            
#             print("Plotting training curves...")
#             plot_training_curves(losses)
            
#             # Save model
#             torch.save(model.state_dict(), 'state_predictor.pth')
#             print("Model saved as 'state_predictor.pth'")
#     else:
#         print("No training data collected. Check if graph states are properly enabled.")
