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
        self.nodes, self.edges = self.build_graph_info()
        if self.use_graph_as_states:
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
                    
                    # Create edges: source -> buffer -> target
                    edges.append({
                        'source': source_node,
                        'target': obj_name,  # buffer name
                        'type': 'feeds_into',
                        'attributes': []  # Can add attributes if needed
                    })
                    
                    edges.append({
                        'source': obj_name,  # buffer name
                        'target': target_node,
                        'type': 'feeds_from',
                        'attributes': []  # Can add attributes if needed
                    })
            else:
                # it's a node
                if type(obj).__name__ == 'WorkerPool':
                    # first connect to the stations, then get the stations status for the observation
                    # workerpool features
                    # [n_workers, station_performance * n_stations]
                    features = [self._objects[obj_name].n_workers, self._objects[obj_name].n_stations, self._objects[obj_name].transition_time]
                    throughput_rates = []
                    connected_stations = self._objects[obj_name]._station_names
                    current_worker_status = self._objects[obj_name].state.values[:self._objects[obj_name].n_workers]
                    _, occurences = np.unique(current_worker_status, return_counts=True)
                    for i,station_name in enumerate(connected_stations):
                        # features from the workerpool
                        features_from_pool = [occurences[i]]  # number of workers assigned to this station
                        features_from_pool = np.array(features_from_pool, dtype=np.float32)
                        # features from the assembly station TODO: check if necessary
                        features_to_get_from_assembly = ['current_window_throughput', 'processing_time', 'n_workers']
                        features_from_assembly = [self._objects[station_name].state[j].value for j in features_to_get_from_assembly]
                        features_from_assembly = np.array(features_from_assembly, dtype=np.float32)
                        # get the throughput rate to calculate the mean throughput rate of connected stations
                        throughput_rates.append(self._objects[station_name].state['current_window_throughput'].value)

                        # for each connected station, create an edge, and save the feature
                        edges.append({
                            'source': obj_name,
                            'target': station_name,
                            'type': 'assigned_to',
                            'attributes': np.array(features_from_pool, dtype=np.float32)
                        })
                        edges.append({
                            'source': station_name,
                            'target': obj_name,
                            'type': 'assigned_from',
                            'attributes': np.array(features_from_assembly, dtype=np.float32)
                        })
                    # Append the mean throughput rate to the features
                    features.append(np.mean(throughput_rates))

                    node_info = {
                        'name': obj_name,
                        'type': type(obj).__name__,
                        'feature': features
                    }
                else:
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
            edge_attrs[edge_type].append(edge_data.get('attributes', []))
            if add_reverse_edges and source_type != 'WorkerPool' and target_type != 'WorkerPool':
                reverse_edge_type = (target_type, 'upstream', source_type)
                if reverse_edge_type not in edge_types:
                    edge_types[reverse_edge_type] = []
                    edge_attrs[reverse_edge_type] = []
                    edge_mapping[reverse_edge_type] = []
                edge_types[reverse_edge_type].append([target_idx, source_idx])
                edge_attrs[reverse_edge_type].append(edge_data.get('attributes', []))
            # edge_mapping[edge_type].append(edge_data.get('buffer', {}))

        # Add self-loop edges for each node type
        # for node_type, node_list in node_types.items():
        #     edge_type = (node_type, 'self_loop', node_type)
        #     if edge_type not in edge_types:
        #         edge_types[edge_type] = []
        #         edge_attrs[edge_type] = []
        #         # edge_mapping[edge_type] = []
        #     for i, (node_name, node_data) in enumerate(node_list):
        #         edge_types[edge_type].append([i, i])
        #         # For self-loop, you can use a default attribute or the node's own feature
        #         # Here, we use zeros as default self-loop attributes
        #         attr = np.zeros_like(node_data)
        #         edge_attrs[edge_type].append(attr)
        #         # edge_mapping[edge_type].append(f"self_loop_{node_name}")
        
        # Add self-loop to only the source nodes
        for node_list in node_types['Source']:
            node_name, node_data = node_list
            edge_type = ('Source', 'self_loop', 'Source')
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
        # extract features to update
        # if not self.use_graph_as_states or not hasattr(self, "_graph_states"):
        #     return

        # update node features
        for node_name, (node_type, node_idx) in self.node_mapping.items():
            obj = self._objects.get(node_name)
            if type(obj).__name__ == 'WorkerPool':
                # special case for WorkerPool, need to update the connected stations performance
                connected_stations = obj._station_names
                features = [obj.n_workers, obj.n_stations, obj.transition_time]
                throughput_rates = []
                for station_name in connected_stations:
                    throughput_rates.append(self._objects[station_name].state['current_window_throughput'].value)
                features.append(np.mean(throughput_rates))
                features = np.array(features, dtype=np.float32)
                new_feature = torch.tensor(features, dtype=torch.float)
                self._graph_states[node_type].x[node_idx] = new_feature
            elif obj and 'Buffer_' in node_name:
                # Handle buffer nodes
                state_values = np.array(obj.state.values, dtype=np.float32)
                observables = np.array(obj.state.observables, dtype=bool)
                filtered_values = state_values[observables]
                new_feature = torch.tensor(filtered_values, dtype=torch.float)
                self._graph_states[node_type].x[node_idx] = new_feature
            elif obj:
                # Regular stations
                state_values = np.array(obj.state.values, dtype=np.float32)
                observables = np.array(obj.state.observables, dtype=bool)
                filtered_values = state_values[observables]
                new_feature = torch.tensor(filtered_values, dtype=torch.float)
                self._graph_states[node_type].x[node_idx] = new_feature
            
        # update edge attributes if any
        for edge_type in self._graph_states.edge_types:
            source_type = edge_type[0]
            link_type = edge_type[1]
            target_type = edge_type[2]
            
            edge_index = self._graph_states[edge_type].edge_index
            if source_type == target_type:
                # self-loop edges
                continue
            if target_type == 'WorkerPool' and source_type == 'Assembly':
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    src_name = self.node_types[source_type][src_idx][0]
                    # features from the assembly station TODO: check if necessary
                    features_to_get_from_assembly = ['current_window_throughput', 'processing_time', 'n_workers']
                    features_from_assembly = [self._objects[src_name].state[j].value for j in features_to_get_from_assembly]
                    features_from_assembly = np.array(features_from_assembly, dtype=np.float32)
                    self._graph_states[edge_type].edge_attr[i] = torch.tensor(features_from_assembly, dtype=torch.float)
                continue
            elif source_type == 'WorkerPool' and target_type == 'Assembly':
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    src_name = self.node_types[source_type][src_idx][0]
                    current_worker_status = self._objects[src_name].state.values[:self._objects[src_name].n_workers]
                    # Count occurrences for all possible numbers (e.g., 0 to n_stations-1)
                    possible_numbers = np.arange(self._objects[src_name].n_stations)
                    occurences = np.array([(current_worker_status == n).sum() for n in possible_numbers])
                    for i in range(edge_index.shape[1]):
                        src_idx = edge_index[0, i].item()
                        src_name = self.node_types[source_type][src_idx][0]
                        # features from the assembly station TODO: check if necessary
                        features_from_pool = [occurences[i]]  # number of workers assigned to this station
                        features_from_pool = np.array(features_from_pool, dtype=np.float32)
                        self._graph_states[edge_type].edge_attr[i] = torch.tensor(features_from_pool, dtype=torch.float)
                continue

            # # edge_index shape: [2, num_edges], so iterate over columns
            # for i in range(edge_index.shape[1]):
            #     src_idx = edge_index[0, i].item()
            #     tgt_idx = edge_index[1, i].item()
            #     src_name = self.node_types[source_type][src_idx][0]
            #     tgt_name = self.node_types[target_type][tgt_idx][0]
            #     # check if tgt_name is Worker
            #     if src_name.startswith("W") and src_name[1:].isdigit():
            #         # src_name looks like "W" followed by a number
            #         continue
            #     if tgt_name.startswith("W") and tgt_name[1:].isdigit():
            #         # tgt_name looks like "W" followed by a number
            #         continue
                
            #     edge_name = f"Buffer_{src_name}_to_{tgt_name}"
            #     if link_type == 'upstream':
            #         edge_name = f"Buffer_{tgt_name}_to_{src_name}"
            #     self._graph_states[edge_type].edge_attr[i] = torch.tensor(self._objects[edge_name].state.values, dtype=torch.float)
                
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
