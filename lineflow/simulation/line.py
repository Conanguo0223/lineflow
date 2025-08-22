import simpy
import pygame
import numpy as np
import logging
from tqdm import tqdm
from torch_geometric.data import HeteroData
import torch
from lineflow.simulation.stationary_objects import StationaryObject
from lineflow.simulation.states import LineStates
from lineflow.simulation.connectors import Connector
from lineflow.simulation.stations import (
    Station,
    Sink,
)

logger = logging.getLogger(__name__)


class Line:
    """
    Args:
        realtime (bool): Only if `visualize` is `True`
        factor (float): visualization speed
        info (list): A list of line data that is retrivable over the get_info() method.
            That is `info = [("A1", n_workers), ("A3", "assembly_time")]`.
            Data will be logged in experiments.
    """

    def __init__(
        self,
        realtime=False,
        factor=0.5,
        random_state=10,
        step_size=1,
        scrap_factor=1,
        use_graph_as_states=False,
        info=None,
    ):

        # TODO: This attribute needs to be refactored in future as it is only used by the
        # gym-simulation
        self.scrap_factor = scrap_factor
        self.realtime = realtime
        self.factor = factor
        self.step_size = step_size
        # graph
        self.use_graph_as_states = use_graph_as_states
        self.node_mapping = None # use for updating the graph features
        self.node_types = None
        self._graph_states = None
        self.nodes = None
        self.edges = None
        if self.use_graph_as_states:
            self._graph_states = HeteroData()
        if info is None:
            info = []
        self._info = info

        self.reset(random_state=random_state)

    @property
    def name(self):
        return self.__class__.__name__

    def info(self):
        """
        Returns additional Information about the line
        """
        general = {
            "name": self.name,
            "T": self.env.now,
            "n_parts": self.get_n_parts_produced(),
            "n_scrap_parts": self.get_n_scrap_parts(),
        }

        additional = {
            f"{station}_{attribute}": self.state.objects[station].states[attribute].value
            for station, attribute in self._info
        }
        return {**general, **additional}

    def _make_env(self):
        if self.realtime:
            self.env = simpy.rt.RealtimeEnvironment(factor=self.factor, strict=False)
        else:
            self.env = simpy.Environment()

    def _make_objects(self):
        """
        Builds the LineObjects
        """
        # Build the stations and connectors
        with StationaryObject() as objects:
            self.build()

        self._objects = {}

        for obj in objects:
            if obj.name in self._objects:
                raise ValueError(f'Multiple objects with name {obj.name} exist')
            self._objects[obj.name] = obj

        # Validate carrier specs
        for obj in self._objects.values():
            if hasattr(obj, 'carrier_specs'):
                self._validate_carrier_specs(obj.carrier_specs)

    def _validate_carrier_specs(self, specs):
        for carrier_name, part_specs in specs.items():
            for part_name, part_spec in part_specs.items():
                for station in part_spec.keys():
                    if station not in self._objects:
                        raise ValueError(
                                f"Spec for part '{part_name}' in carrier '{carrier_name}' "
                                f"contains unkown station '{station}'"
                        )

    def _build_states(self):
        """
        Builds the states of the line objects as well as the LineState
        """
        object_states = {}

        for name, obj in self._objects.items():
            obj.init(self.random)
            object_states[name] = obj.state

        self.state = LineStates(object_states, self.env)
        
        if self.use_graph_as_states:
            self.nodes, self.edges = self.build_graph_info()
            self._graph_states = self.build_graph_state(self.nodes, self.edges)

    def build_graph_info(self):
        """
        Builds the graph representation of the line
        """
        nodes = {}
        edges = []

        for obj_name in self.state.object_names:
            obj = self._objects.get(obj_name)
            
            # check if the object is an edge or node
            # edge needs edge_index, and edge attributes
            if 'Buffer_' in obj_name and '_to_' in obj_name:
                # Parse buffer name: Buffer_Source_to_Assembly
                parts = obj_name.replace('Buffer_', '').split('_to_')
                if len(parts) != 2:
                    raise ValueError(f"Invalid buffer name: {obj_name}")
                else:
                    source_node, target_node = parts
                    
                                            
                    # Add buffer as a node
                    features = self._objects[obj_name].state.values
                    observables = self._objects[obj_name].state.observables
                    # Only keep observables
                    features = np.array(features, dtype=np.float32)[observables]
                    
                    node_info = {
                        'name': obj_name,
                        'type': 'Buffer',
                        'feature': features,
                        'connects_from': source_node,
                        'connects_to': target_node
                    }
                    nodes[obj_name] = node_info
                    
                    # Check source and target types
                    source_is_switch = type(self._objects[source_node]).__name__ == 'Switch'
                    target_is_switch = type(self._objects[target_node]).__name__ == 'Switch'
                    
                    # Case 1: Switch → Buffer → Switch
                    if source_is_switch and target_is_switch:
                        source_switch = self._objects[source_node]
                        target_switch = self._objects[target_node]
                        
                        # Get usage status from source switch (output buffer)
                        source_usage_status = self._get_switch_buffer_usage_status(
                            source_switch, obj_name, 'output'
                        )
                        
                        # Get usage status from target switch (input buffer)
                        target_usage_status = self._get_switch_buffer_usage_status(
                            target_switch, obj_name, 'input'
                        )
                        
                        edges.append({
                            'source': source_node,
                            'target': obj_name,
                            'type': 'switchesinto',
                            'attributes': [source_usage_status]
                        })
                        
                        edges.append({
                            'source': obj_name,
                            'target': target_node,
                            'type': 'switchesfrom',
                            'attributes': [target_usage_status]
                        })
                    
                    # Case 2: Switch → Buffer → Other
                    elif source_is_switch and not target_is_switch:
                        switch_obj = self._objects[source_node]
                        buffer_usage_status = self._get_switch_buffer_usage_status(
                            switch_obj, obj_name, 'output'
                        )
                        
                        edges.append({
                            'source': source_node,
                            'target': obj_name,
                            'type': 'switchesinto',
                            'attributes': [buffer_usage_status]
                        })
                        
                        edges.append({
                            'source': obj_name,
                            'target': target_node,
                            'type': 'feedsfrom',
                            'attributes': []  # Regular connection to non-switch
                        })
                    
                    # Case 3: Other → Buffer → Switch
                    elif not source_is_switch and target_is_switch:
                        switch_obj = self._objects[target_node]
                        buffer_usage_status = self._get_switch_buffer_usage_status(
                            switch_obj, obj_name, 'input'
                        )
                        
                        edges.append({
                            'source': source_node,
                            'target': obj_name,
                            'type': 'feedsinto',
                            'attributes': []  # Regular connection from non-switch
                        })
                        
                        edges.append({
                            'source': obj_name,
                            'target': target_node,
                            'type': 'switchesfrom',
                            'attributes': [buffer_usage_status]
                        })
                    
                    # Case 4: Other → Buffer → Other (no switch involved)
                    else:
                        edges.append({
                            'source': source_node,
                            'target': obj_name,
                            'type': 'feedsinto',
                            'attributes': []
                        })
                        
                        edges.append({
                            'source': obj_name,
                            'target': target_node,
                            'type': 'feedsfrom',
                            'attributes': []
                        })
            else:
                # it's a node
                if type(obj).__name__ == 'WorkerPool':
                    # first connect to the stations, then get the stations status for the observation
                    # workerpool features get the station features
                    features = self._objects[obj_name].state.values
                    observables = self._objects[obj_name].state.observables
                    features = np.array(features, dtype=np.float32)[observables]
                    
                    # get the station and connect them
                    available_stations = self._objects[obj_name]._station_names
                    current_worker_status = self._objects[obj_name].state.values[:self._objects[obj_name].n_workers]
                    _, occurences = np.unique(current_worker_status, return_counts=True)
                    # create the node for workerpool
                    node_info = {
                        'name': obj_name,
                        'type': type(obj).__name__,
                        'feature': features
                    }

                    for i,station_name in enumerate(available_stations):
                        # features from the workerpool
                        is_assigned = occurences[i] > 0
                        assign_ratio = occurences[i] / self._objects[obj_name].n_workers if self._objects[obj_name].n_workers > 0 else 0

                        features_from_pool = [is_assigned, assign_ratio]
                        features_from_pool = np.array(features_from_pool, dtype=np.float32)
                        
                        # features from the station
                        features_from_assembly = self._objects[station_name]._get_edge_features_to_pool()

                        # TODO: check if in the future, the connected node related features should be included
                        # since current design is bidirectional, I'll omit it

                        # for each connected station, create an edge, and save the feature
                        edges.append({
                            'source': obj_name,
                            'target': station_name,
                            'type': 'assignedto',
                            'attributes': np.array(features_from_pool, dtype=np.float32)
                        })
                        edges.append({
                            'source': station_name,
                            'target': obj_name,
                            'type': 'assignedfrom',
                            'attributes': np.array(features_from_assembly, dtype=np.float32)
                        })
                else:
                    # if not a workerpool
                    features = self._objects[obj_name].state.values
                    observables = self._objects[obj_name].state.observables
                    # Only keep observables
                    features = np.array(features, dtype=np.float32)[observables]
                    node_info = {
                        'name': obj_name,
                        'type': type(obj).__name__,
                        'feature': features
                    }

                nodes[obj_name] = node_info
        return nodes, edges
    def _get_switch_buffer_usage_status(self, switch_obj, buffer_name, buffer_type):
        """
        Check if a buffer connected to a switch is currently being used
        
        Args:
            switch_obj: The Switch station object
            buffer_name: Name of the buffer (e.g., "Buffer_Switch_to_Station1")
            buffer_type: Either 'input' or 'output'
        
        Returns:
            float: 1.0 if buffer is currently selected/active, 0.0 otherwise
        """
        try:
            if buffer_type == 'input':
                # Check if this buffer is the currently selected input buffer
                if hasattr(switch_obj, 'buffer_in') and hasattr(switch_obj.state, 'index_buffer_in'):
                    current_input_index = switch_obj.state['index_buffer_in'].value
                    
                    # Get all input buffer names to find the index of our buffer
                    input_buffer_names = []
                    if isinstance(switch_obj.buffer_in, list):
                        for buffer_method in switch_obj.buffer_in:
                            # Extract buffer name from method
                            buffer_obj = buffer_method.__self__
                            input_buffer_names.append(buffer_obj.name)
                    
                    # Check if our buffer is the currently selected one
                    if buffer_name in input_buffer_names:
                        buffer_index = input_buffer_names.index(buffer_name)
                        return 1.0 if buffer_index == current_input_index else 0.0
                        
            elif buffer_type == 'output':
                # Check if this buffer is the currently selected output buffer
                if hasattr(switch_obj, 'buffer_out') and hasattr(switch_obj.state, 'index_buffer_out'):
                    current_output_index = switch_obj.state['index_buffer_out'].value
                    
                    # Get all output buffer names to find the index of our buffer
                    output_buffer_names = []
                    if isinstance(switch_obj.buffer_out, list):
                        for buffer_method in switch_obj.buffer_out:
                            # Extract buffer name from method
                            buffer_obj = buffer_method.__self__
                            output_buffer_names.append(buffer_obj.name)
                    
                    # Check if our buffer is the currently selected one
                    if buffer_name in output_buffer_names:
                        buffer_index = output_buffer_names.index(buffer_name)
                        return 1.0 if buffer_index == current_output_index else 0.0
            
            # Alternative method: Parse buffer name to determine which buffer it is
            # If buffer name is "Buffer_Switch_to_Station1", extract "Station1"
            if '_to_' in buffer_name:
                connected_station = buffer_name.split('_to_')[-1]
                
                # For output buffers: check if switch is currently routing to this station
                if buffer_type == 'output' and hasattr(switch_obj, '_station_names'):
                    if hasattr(switch_obj.state, 'index_buffer_out'):
                        current_index = switch_obj.state['index_buffer_out'].value
                        station_names = switch_obj._station_names if hasattr(switch_obj, '_station_names') else []
                        
                        if current_index < len(station_names):
                            current_target = station_names[current_index]
                            return 1.0 if current_target == connected_station else 0.0
            
            # If we can't determine the status, return 0.0 (not active)
            return 0.0
            
        except Exception as e:
            print(f"Warning: Could not determine buffer usage status for {buffer_name}: {e}")
            return 0.0
        
    def build_graph_state(self, nodes, edges):
        """
        Builds the graph state from nodes and edges information,
        including self-loop edges for each node.
        """
        graph_state = HeteroData()

        # Group up the nodes according to their type
        node_types = {}
        for node_name, node_info in nodes.items():
            node_type = node_info['type']
            node_data = node_info['feature']
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append((node_name, node_data))
        self.node_types = node_types
        # Create node features and mappings
        node_mapping = {}
        for node_type, node_list in node_types.items():
            features = []
            for i, (node_name, node_data) in enumerate(node_list):
                features.append(node_data)
                node_mapping[node_name] = (node_type, i)
            features = np.array(features)
            graph_state[node_type].x = torch.tensor(features, dtype=torch.float)
        self.node_mapping = node_mapping

        # Group and add edges
        edge_types = {}
        edge_attrs = {}
        edge_mapping = {}
        add_reverse_edges = True  # Add reverse edges for directed edges
        for edge_data in edges:
            source_name = edge_data['source']
            target_name = edge_data['target']
            source_type = node_mapping[source_name][0]
            target_type = node_mapping[target_name][0]
            edge_type = (source_type, edge_data['type'], target_type)
            if edge_type not in edge_types:
                edge_types[edge_type] = []
                edge_attrs[edge_type] = []
                edge_mapping[edge_type] = []
            source_idx = node_mapping[source_name][1]
            target_idx = node_mapping[target_name][1]
            edge_types[edge_type].append([source_idx, target_idx])
            edge_attrs[edge_type].append(edge_data.get('attributes'))
            if add_reverse_edges and source_type != 'WorkerPool' and target_type != 'WorkerPool':
                reverse_edge_type = (target_type, 'upstream', source_type)
                if reverse_edge_type not in edge_types:
                    edge_types[reverse_edge_type] = []
                    edge_attrs[reverse_edge_type] = []
                    edge_mapping[reverse_edge_type] = []
                edge_types[reverse_edge_type].append([target_idx, source_idx])
                edge_attrs[reverse_edge_type].append(edge_data.get('attributes'))
            # edge_mapping[edge_type].append(edge_data.get('buffer', {}))
        
        # Add self-loop to only the source nodes
        for node_list in node_types['Source']:
            node_name, node_data = node_list
            edge_type = ('Source', 'selfloop', 'Source')
            if edge_type not in edge_types:
                edge_types[edge_type] = []
                edge_attrs[edge_type] = []
                # edge_mapping[edge_type] = []
            source_idx = node_mapping[node_name][1]
            edge_types[edge_type].append([source_idx, source_idx])
            # For self-loop, you can use a default attribute or the node's own feature
            # Here, we use zeros as default self-loop attributes
            attr = np.zeros_like(node_data)
            edge_attrs[edge_type].append(attr)
            # edge_mapping[edge_type].append(f"self_loop_{node_name}")

        # Add edges and edge attributes to HeteroData
        for edge_type, edge_list in edge_types.items():
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                graph_state[edge_type].edge_index = edge_index
            attrs = edge_attrs[edge_type]
            if attrs and all(isinstance(a, (list, np.ndarray, torch.Tensor)) for a in attrs):
                edge_attr_tensor = torch.tensor(np.array(attrs), dtype=torch.float)
                graph_state[edge_type].edge_attr = edge_attr_tensor
            else:
                pass

        return graph_state

    def reset(self, random_state=None):
        """
        Resets the simulation.
        """
        self.random = np.random.RandomState(random_state)
        self._make_env()
        self._make_objects()

        self._build_states()
        self._register_objects_at_env()

        self.end_step = 0
        self.env.process(self.step_event())

    def _assert_one_sink(self):
        if len([c for c in self._objects.values() if isinstance(c, Sink)]) != 1:
            raise ValueError(
                "Number of sinks does not match"
                "Currently, only scenarios with exactly one sink are allowed"
            )

    def get_sink(self):
        sinks = [s for s in self._objects.values() if isinstance(s, Sink)]
        self._assert_one_sink()
        return sinks[0]

    def get_n_scrap_parts(self):
        """
        Returns the number of produced parts up to now
        """
        return self.state.get_n_scrap_parts()

    def get_n_parts_produced(self):
        """
        Returns the number of produced parts up to now
        """
        return self.state.get_n_parts_produced()
    
    def get_uptime(self, lookback=None):
        """
        Returns the uptime of the line 
        """
        return self.state.get_uptime(lookback=lookback)

    def build(self):
        """
        This function should add objects of the LineObject class as attributes
        """
        raise NotImplementedError()

    def _register_objects_at_env(self):
        """
        Registers all line objects at the simpy simulation environment.
        """
        for o in self._objects.values():
            o.register(self.env)

    def _draw(self, screen, actions=None):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        screen.fill('white')

        font = pygame.font.SysFont(None, 20)

        time = font.render('T={:.2f}'.format(self.env.now), True, 'black')
        n_parts = font.render(
            f'#Parts={self.get_n_parts_produced()}', True, 'black'
        )

        screen.blit(time, time.get_rect(center=(30, 30)))
        screen.blit(n_parts, n_parts.get_rect(center=(30, 50)))

        # Draw objects, first connectors, then stations
        self._draw_connectors(screen)
        self._draw_stations(screen)
        if actions:
            self._draw_actions(screen, actions)
        pygame.display.flip()

    def _draw_actions(self, screen, actions):
        font = pygame.font.SysFont(None, 20)
        actions = font.render(f'{actions}', True, 'black')
        screen.blit(actions, actions.get_rect(center=(500, 30)))
        pygame.display.flip()

    def _draw_stations(self, screen):
        self._draw_objects_of_type(screen, Station)

    def _draw_connectors(self, screen):
        self._draw_objects_of_type(screen, Connector)

    def _draw_objects_of_type(self, screen, object_type):
        for name, obj in self._objects.items():
            if isinstance(obj, object_type):
                obj._draw(screen)

    def setup_draw(self):
        pygame.init()
        x = []
        y = []
        for o in self._objects.values():
            o.setup_draw()
            if isinstance(o, Station):
                assert hasattr(o, "position"), f"Please provide position for {Station.name}"
                x.append(o.position[0])
                y.append(o.position[1])

        return pygame.display.set_mode((max(x) + 100, max(y) + 100))

    def teardown_draw(self):
        pygame.quit()

    def apply(self, values):
        for object_name in values.keys():
            self._objects[object_name].apply(values[object_name])

    def step(self, simulation_end=None):
        """
        Step to the next state of the line
        Args:
            simulation_end (int):
                Time until terminated flag is returned as True. If None
                terminated is always False.
        """
        terminated = False

        # The end of the the current step, excluding the event execution
        # i.e. execute all events where scheudled_time < end_step
        self.end_step = self.end_step + self.step_size

        while True:
            if self.env.peek() > self.end_step:
                self.state.log()
                # If the next event is scheduled after simulation end
                if simulation_end is not None and self.env.peek() > simulation_end:
                    terminated = True
                if self.use_graph_as_states:
                    self.update_graph_state()
                    return self._graph_states, terminated
                return self.state, terminated

                # self.state.log()
                # # If the next event is scheduled after simulation end
                # if simulation_end is not None and self.env.peek() > simulation_end:
                #     terminated = True

                # return self.state, terminated

            self.env.step()

    def update_graph_state(self):
        """
        Update graph state including all edge types from build_graph_info
        """
        # Update node features
        for node_name, (node_type, node_idx) in self.node_mapping.items():
            obj = self._objects.get(node_name)
            state_values = np.array(obj.state.values, dtype=np.float32)
            observables = np.array(obj.state.observables, dtype=bool)
            filtered_values = state_values[observables]
            new_feature = torch.tensor(filtered_values, dtype=torch.float)
            self._graph_states[node_type].x[node_idx] = new_feature
        
        # Update edge attributes for all edge types
        for edge_type in self._graph_states.edge_types:
            source_type = edge_type[0]
            link_type = edge_type[1]
            target_type = edge_type[2]
            
            edge_index = self._graph_states[edge_type].edge_index
            
            if source_type == target_type:
                # Self-loop edges - skip
                continue
            
            # Handle WorkerPool edges
            if target_type == 'WorkerPool' and source_type in ['Assembly', 'Process']:
                # Station → WorkerPool edges (assigned_from)
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    src_name = self.node_types[source_type][src_idx][0]
                    
                    # Get updated features from the station
                    features_from_station = self._objects[src_name]._get_edge_features_to_pool()
                    self._graph_states[edge_type].edge_attr[i] = torch.tensor(features_from_station, dtype=torch.float)
                continue
                
            elif source_type == 'WorkerPool' and target_type in ['Assembly', 'Process']:
                # WorkerPool → Station edges (assigned_to)
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    tgt_idx = edge_index[1, i].item()
                    src_name = self.node_types[source_type][src_idx][0]
                    tgt_name = self.node_types[target_type][tgt_idx][0]
                    
                    # Get current worker assignments
                    current_worker_status = self._objects[src_name].state.values[:self._objects[src_name].n_workers]
                    possible_numbers = np.arange(self._objects[src_name].n_stations)
                    occurences = np.array([(current_worker_status == n).sum() for n in possible_numbers])
                    
                    # Find which station index this edge corresponds to
                    available_stations = self._objects[src_name]._station_names
                    station_idx = available_stations.index(tgt_name)
                    
                    # Calculate features for this specific edge
                    is_assigned = occurences[station_idx] > 0
                    assign_ratio = occurences[station_idx] / self._objects[src_name].n_workers if self._objects[src_name].n_workers > 0 else 0
                    
                    features_from_pool = [is_assigned, assign_ratio]
                    features_from_pool = np.array(features_from_pool, dtype=np.float32)
                    self._graph_states[edge_type].edge_attr[i] = torch.tensor(features_from_pool, dtype=torch.float)
                continue
            
            # Handle Switch-related edges
            elif link_type in ['switchesinto', 'switchesfrom']:
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    tgt_idx = edge_index[1, i].item()
                    src_name = self.node_types[source_type][src_idx][0]
                    tgt_name = self.node_types[target_type][tgt_idx][0]
                    
                    # Determine which node is the switch and which is the buffer
                    if type(self._objects[src_name]).__name__ == 'Switch':
                        switch_obj = self._objects[src_name]
                        buffer_name = tgt_name
                        buffer_type = 'output'
                    elif type(self._objects[tgt_name]).__name__ == 'Switch':
                        switch_obj = self._objects[tgt_name]
                        buffer_name = src_name
                        buffer_type = 'input'
                    else:
                        # This shouldn't happen for switches_into/switches_from edges
                        continue
                    
                    # Get current buffer usage status
                    usage_status = self._get_switch_buffer_usage_status(
                        switch_obj, buffer_name, buffer_type
                    )
                    
                    # Update edge attribute
                    self._graph_states[edge_type].edge_attr[i] = torch.tensor([usage_status], dtype=torch.float)
                continue
            
            # Handle regular buffer edges (feeds_into, feeds_from, upstream)
            elif link_type in ['feedsinto', 'feedsfrom', 'upstream']:
                # These edges typically don't have attributes or have empty attributes
                # If you want to add buffer state as edge attributes, you can do:
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    tgt_idx = edge_index[1, i].item()
                    src_name = self.node_types[source_type][src_idx][0]
                    tgt_name = self.node_types[target_type][tgt_idx][0]
                    
                    # For buffer edges, you might want to include buffer fill state
                    if 'Buffer_' in src_name:
                        buffer_obj = self._objects[src_name]
                        fill_state = buffer_obj.get_fillstate() if hasattr(buffer_obj, 'get_fillstate') else 0.0
                        self._graph_states[edge_type].edge_attr[i] = torch.tensor([fill_state], dtype=torch.float)
                    elif 'Buffer_' in tgt_name:
                        buffer_obj = self._objects[tgt_name]
                        fill_state = buffer_obj.get_fillstate() if hasattr(buffer_obj, 'get_fillstate') else 0.0
                        self._graph_states[edge_type].edge_attr[i] = torch.tensor([fill_state], dtype=torch.float)
                    # If no buffer involved, keep empty attributes (or add other relevant info)
                continue
    def step_event(self):
        """
        Ensures that there is an Event scheduled for `self.step_size` intervals
        The step function is only able to stop the simulation if an Event is scheduled.
        """
        while True:
            yield self.env.timeout(self.step_size)
    def run(
        self,
        simulation_end,
        agent=None,
        show_status=True,
        visualize=False,
        capture_screen=False,
        collect_data = False,
        # record_states = False,
    ):
        """
        Args:
            simulation_end (float): Time until the simulation stops
            agent (lineflow.models.reinforcement_learning.agents): An Agent that interacts with a
                line. Can also be just a policy if an __call__ method exists like in the BaseAgent
                class.
            show_status (bool): Show progress bar for each simulation episode
            visualize (bool): If true, line visualization is opened
            capture_screen (bool): Captures last Time frame when screen should be recorded
        """
        # all_states = None
        # if record_states:
        #     all_states = []
        if visualize:
            # Stations first, then connectors
            screen = self.setup_draw()

        # Register objects when simulation is initially started
        if len(self.env._queue) == 0:
            self._register_objects_at_env()

        # # TEST data collection
        # self.env.process(self.collect_station_info_process(interval=10))

        now = 0
        actions = None
        pbar = tqdm(
            total=simulation_end,
            bar_format='{desc}: {percentage:3.2f}%|{bar:50}|',
            disable=not show_status,
        )
        if collect_data:
            collected_states = []
            assert self.use_graph_as_states is True, "need to use graph states for data collection"
        while self.env.now < simulation_end:
            pbar.update(self.env.now - now)
            now = self.env.now
            try:
                # Step the simulation
                state, terminated = self.step(simulation_end=simulation_end)
                # if all_states is not None:
                #     all_states.append(state)
                # print(state)
                # Collect the current graph state (don't overwrite self._graph_states!)
                if collect_data:
                    if self._graph_states is not None:
                        # Clone the HeteroData to avoid reference issues
                        collected_states.append(self._graph_states.clone())
                    else:
                        print(f"Warning: _graph_states is None at time {self.env.now}")
            except simpy.core.EmptySchedule:
                logger.warning('Simulation in dead-lock - end early')
                break

            if agent is not None:
                actions = agent(self.state, self.env)
                self.apply(actions)

            if visualize:
                if actions is not None:
                    self._draw(screen, actions)
                else:
                    self._draw(screen)
        if capture_screen and visualize:
            pygame.image.save(screen, f"{self.name}.png")

        if visualize:
            self.teardown_draw()
        # return the states
        # if collect_data and record_states:
        #     return collected_states, all_states
        elif collect_data:
            return collected_states
        # elif record_states:
        #     return all_states
        
    def get_observations(self, object_name=None):
        """
        """

        df = self.state.df()

        if object_name is None:
            return df
        else:
            cols = [c for c in df.columns if c.startswith(object_name)]
            cols = cols + ['T_start', 'T_end']
            return df[cols].rename(
                columns={
                    c: c.replace(object_name + '_', '') for c in cols
                }
            )

    def __getitem__(self, name):
        return self._objects[name]

    def collect_station_info_process(self, interval=10):
        """
        SimPy process that collects info from all stations every `interval` time units.
        """
        collected_info = []
        while True:
            # Collect info from all Station objects
            info = {}
            for name, obj in self._objects.items():
                if isinstance(obj, Station):
                    info[name] = obj.state.values  # or obj.state.df(), etc.
            collected_info.append((self.env.now, info))
            yield self.env.timeout(interval)
        # Optionally, return collected_info at the end (if you want to access it later)
        # return collected_info
