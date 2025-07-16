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
                    edges.append({
                        'source': source_node,
                        'target': target_node,
                        'buffer': obj_name,
                        'type': 'connects_to',
                        'attributes': self._objects[obj_name].state.values
                    })
            else:
                # it's a node
                if type(obj).__name__ == 'WorkerPool':
                    # WorkerPool is a special case, we want to treat it as a nod
                    node_info = {
                        'name': obj_name,
                        'type': type(obj).__name__,
                        'feature': self._objects[obj_name].state.values
                    }
                    for worker_name, worker in obj.workers.items():
                        nodes[worker_name] = {
                            'name': worker_name,
                            'type': 'Worker',
                            'feature': [worker.state.value, worker.transition_time]
                        }
                        
                        # Connect pool to workers
                        edges.append({
                            'source': obj_name,
                            'target': worker_name,
                            'type': 'manages'
                        })
                        
                        # Connect workers to their assigned stations
                        assigned_station = obj.stations[worker.state.value].name
                        edges.append({
                            'source': worker_name,
                            'target': assigned_station,
                            'type': 'assigned_to'
                        })
                else:
                    node_info = {
                        'name': obj_name,
                        'type': type(obj).__name__,
                        'feature': self._objects[obj_name].state.values
                    }
                    
                    node_info['node_properties'] = self._extract_component_properties(obj)

                nodes[obj_name] = node_info
        return nodes, edges


    def _extract_component_properties(self, component):
        """Extract component-specific properties"""
        properties = {}
        
        # Check component type and extract relevant properties
        component_type = type(component).__name__
        # type_lookup = {
        #     'Assembly': 1,
        #     'Process': 2,
        #     'Source': 3,
        #     'Sink': 4,
        #     'Switch': 5,
        #     'Magazine': 6,
        # }
        # properties['type_id'] = type_lookup.get(component_type, 0)

        if component_type == 'Assembly':
            properties.update({
                'NOK_part_error_time': getattr(component, 'NOK_part_error_time', None)
            })
        elif component_type == 'Process':
            properties.update({
                'rework_probability': getattr(component, 'rework_probability', None),
                # 'worker_pool': getattr(component, 'worker_pool', None) # TODO: manage the connection of worker pools
            })
        elif component_type == 'Source':
            properties.update({
                'unlimited_carriers': getattr(component, 'unlimited_carriers', False),
                'carrier_capacity': getattr(component, 'carrier_capacity', None),
                'processing_time': getattr(component, 'processing_time', None)
            })
        elif component_type == 'Sink':
            properties.update({
                'scrap_factor': getattr(component, 'scrap_factor', None),
                'processing_time': getattr(component, 'processing_time', None)
            })
        elif component_type == 'Switch':
            properties.update({
                'scrap_factor': getattr(component, 'scrap_factor', None),
                'processing_time': getattr(component, 'processing_time', None)
            })
        elif component_type == 'Magazine':
            properties.update({
                'is_assembly': True,
                'NOK_part_error_time': getattr(component, 'NOK_part_error_time', None)
            })
        
        # Check for actionable properties
        if hasattr(component, 'state') and component.state is not None:
            actionable_states = []
            for state_name, state in component.state.states.items():
                if state.is_actionable:
                    actionable_states.append(state_name)
            properties['actionable_states'] = actionable_states
            properties['controllable'] = len(actionable_states) > 0
        
        return properties
    
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
            edge_mapping[edge_type].append(edge_data.get('buffer', {}))

        # Add self-loop edges for each node type
        for node_type, node_list in node_types.items():
            edge_type = (node_type, 'self_loop', node_type)
            if edge_type not in edge_types:
                edge_types[edge_type] = []
                edge_attrs[edge_type] = []
                edge_mapping[edge_type] = []
            for i, (node_name, node_data) in enumerate(node_list):
                edge_types[edge_type].append([i, i])
                # For self-loop, you can use a default attribute or the node's own feature
                # Here, we use zeros as default self-loop attributes
                attr = np.zeros_like(node_data)
                edge_attrs[edge_type].append(attr)
                edge_mapping[edge_type].append(f"self_loop_{node_name}")

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

                return self.state, terminated

            self.env.step()

    def update_graph_state(self):
        # extract features to update
        # if not self.use_graph_as_states or not hasattr(self, "_graph_states"):
        #     return

        # update node features
        for node_name, (node_type, node_idx) in self.node_mapping.items():
            obj = self._objects.get(node_name)
            if obj:
                new_feature = torch.tensor(obj.state.values, dtype=torch.float)
                self._graph_states[node_type].x[node_idx] = new_feature
        # update edge attributes if any
        for edge_type in self._graph_states.edge_types:
            source_type = edge_type[0]
            target_type = edge_type[2]
            
            edge_index = self._graph_states[edge_type].edge_index
            if source_type == target_type:
                # self-loop edges
                continue
            # edge_index shape: [2, num_edges], so iterate over columns
            for i in range(edge_index.shape[1]):
                src_idx = edge_index[0, i].item()
                tgt_idx = edge_index[1, i].item()
                src_name = self.node_types[source_type][src_idx][0]
                tgt_name = self.node_types[target_type][tgt_idx][0]
                # check if tgt_name is Worker
                if src_name.startswith("W") and src_name[1:].isdigit():
                    # src_name looks like "W" followed by a number
                    continue
                if tgt_name.startswith("W") and tgt_name[1:].isdigit():
                    # tgt_name looks like "W" followed by a number
                    continue
                
                edge_name = f"Buffer_{src_name}_to_{tgt_name}"
                self._graph_states[edge_type].edge_attr[i] = torch.tensor(self._objects[edge_name].state.values, dtype=torch.float)
                
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

        if visualize:
            # Stations first, then connectors
            screen = self.setup_draw()

        # Register objects when simulation is initially started
        if len(self.env._queue) == 0:
            self._register_objects_at_env()

        now = 0
        actions = None
        pbar = tqdm(
            total=simulation_end,
            bar_format='{desc}: {percentage:3.2f}%|{bar:50}|',
            disable=not show_status,
        )

        while self.env.now < simulation_end:
            pbar.update(self.env.now - now)
            now = self.env.now
            try:
                self.step()
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
