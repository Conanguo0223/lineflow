"""
This file is a wrapper around our simulation such that it is compatible with the stable-baselines
repo
"""
from itertools import filterfalse
import simpy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
from torch_geometric.data import HeteroData
from lineflow.simulation.states import (
    DiscreteState,
    NumericState,
)


def _is_state_supported(state):
    return isinstance(state, DiscreteState) or isinstance(state, NumericState)


def _get_bounds(line_state, state_types='actions', graph=False):
    if state_types not in ["actions", "observations"]:
        raise ValueError('Inconsistent choice')

    
    if graph:
        # if there are states that needs to be tracked
        # we will save the bounds in a dictionary, since there could be different shapes for the states
        state_dict = {}
    else:
        bounds_l = []
        bounds_h = []
    for line_object, state_name in line_state:

        state = line_state[line_object][state_name]

        if (
            (state_types == 'actions' and state.is_actionable) or
            (state_types == 'observations' and state.is_observable)
        ):
            if graph:
                # track new object
                if line_object not in state_dict:
                    state_dict[line_object] = {}
                    state_dict[line_object]['bound_l'] = []
                    state_dict[line_object]['bound_h'] = []

            if not _is_state_supported(state):
                raise ValueError('State not (yet) supported')

            if isinstance(state, DiscreteState):
                if graph:
                    state_dict[line_object]['bound_l'].append(min(state.values))
                    state_dict[line_object]['bound_h'].append(max(state.values))
                else:
                    bounds_l.append(min(state.values))
                    bounds_h.append(max(state.values))
            if isinstance(state, NumericState):
                if state_types == "actions":
                    raise ValueError('Only discrete actions supported')
                if graph:
                    state_dict[line_object]['bound_l'].append(state.vmin)
                    state_dict[line_object]['bound_h'].append(state.vmax)
                else:
                    bounds_l.append(state.vmin)
                    bounds_h.append(state.vmax)
    
    if graph:
        # if there are states that needs to be tracked
        # we will save the bounds in a dictionary, since there could be different shapes for the states
        for line_object in state_dict:
            state_dict[line_object]['bound_l'] = np.array(state_dict[line_object]['bound_l'], dtype=np.float32)
            state_dict[line_object]['bound_h'] = np.array(state_dict[line_object]['bound_h'], dtype=np.float32)
        return state_dict
    else:
        # if there are no states that needs to be tracked
        # we will return the bounds as arrays
        return np.array(bounds_l, dtype=np.float32), np.array(bounds_h, dtype=np.float32)


def _build_observation_space(line_state):
    bounds_l, bounds_h = _get_bounds(line_state, state_types='observations')
    return spaces.Box(
        low=bounds_l.reshape(1, -1),
        high=bounds_h.reshape(1, -1),
    )


def _build_action_space(line_state):
    bounds_l, bounds_h = _get_bounds(line_state, state_types='actions')

    return spaces.MultiDiscrete(
        nvec=bounds_h-bounds_l+1,
        start=bounds_l,
    )

def _build_observation_space_graph(line_state):
    # get the observation space for the graph
    observation_space = spaces.Dict({})
    observation_spaces = _get_bounds(line_state, state_types='observations',graph=True)
    for line_object in observation_spaces:
        observation_space[line_object] = spaces.Box(
            low=observation_spaces[line_object]['bound_l'].reshape(1, -1),
            high=observation_spaces[line_object]['bound_h'].reshape(1, -1),
        )
    return observation_space

def _convert_state_to_hetero_graph(state):
    """
    Convert line state to HeteroData object
    
    Args:
        state: Line state object or graph state from line simulation
        
    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph
    """
    # If state is already a HeteroData object, return it
    if isinstance(state, HeteroData):
        return state
    
    # If state has a graph attribute that's already HeteroData
    if hasattr(state, 'graph') and isinstance(state.graph, HeteroData):
        return state.graph
    
    # Create new HeteroData object
    hetero_data = HeteroData()
    
    # Case 1: State is a dictionary with graph structure
    if hasattr(state, 'items') and callable(getattr(state, 'items')):
        # Get observations as dictionary by object name
        observations = {}
        
        for obj_name, obj_state in state.items():
            if hasattr(obj_state, 'get_observations'):
                obs = obj_state.get_observations(lookback=1, include_time=False)
                observations[obj_name] = np.array(obs, dtype=np.float32)
            else:
                # Fallback: extract observable features manually
                obs = []
                for feature_name, feature in obj_state.items():
                    if hasattr(feature, 'is_observable') and feature.is_observable:
                        obs.append(feature.value)
                observations[obj_name] = np.array(obs, dtype=np.float32)
        
        # Convert observations to node features by object type
        node_type_features = {}
        
        for obj_name, features in observations.items():
            # Get object type from line object
            if hasattr(state, '_line') and hasattr(state._line, '_objects'):
                obj_type = state._line._objects[obj_name].__class__.__name__
            else:
                # Fallback: use object name as type
                obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
            
            # Convert features to tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            
            # Ensure features have correct shape [num_nodes, num_features]
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Add node dimension
            
            # Collect features by node type
            if obj_type not in node_type_features:
                node_type_features[obj_type] = []
            node_type_features[obj_type].append(features)
        
        # Stack features for each node type
        for node_type, feature_list in node_type_features.items():
            hetero_data[node_type].x = torch.cat(feature_list, dim=0)
    
    # Case 2: State already contains x_dict and edge_index_dict
    elif hasattr(state, 'x_dict') and hasattr(state, 'edge_index_dict'):
        # Copy node features
        for node_type, node_features in state.x_dict.items():
            if not isinstance(node_features, torch.Tensor):
                node_features = torch.tensor(node_features, dtype=torch.float32)
            hetero_data[node_type].x = node_features
        
        # Copy edge indices
        for edge_type, edge_index in state.edge_index_dict.items():
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            hetero_data[edge_type].edge_index = edge_index
    
    # Case 3: State is from your graph-enabled line simulation
    elif hasattr(state, 'node_types') and hasattr(state, 'edge_types'):
        # Direct copy from existing HeteroData-like structure
        for node_type in state.node_types:
            if hasattr(state, node_type):
                node_data = getattr(state, node_type)
                if hasattr(node_data, 'x'):
                    hetero_data[node_type].x = node_data.x
        
        for edge_type in state.edge_types:
            if hasattr(state, edge_type):
                edge_data = getattr(state, edge_type)
                if hasattr(edge_data, 'edge_index'):
                    hetero_data[edge_type].edge_index = edge_data.edge_index
    
    return hetero_data


def _convert_hetero_graph_to_dict(hetero_data):
    """
    Convert HeteroData back to dictionary format for observation space compatibility
    
    Args:
        hetero_data: HeteroData object
        
    Returns:
        dict: Dictionary with node and edge information
    """
    obs_dict = {}
    
    # Convert node features to dictionary format
    for node_type in hetero_data.node_types:
        if node_type in hetero_data.x_dict:
            node_features = hetero_data.x_dict[node_type]
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.detach().cpu().numpy()
            obs_dict[f"{node_type}_x"] = node_features.astype(np.float32)
    
    # Convert edge indices to dictionary format  
    for edge_type in hetero_data.edge_types:
        if edge_type in hetero_data.edge_index_dict:
            edge_index = hetero_data.edge_index_dict[edge_type]
            if isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.detach().cpu().numpy()
            
            # Create readable edge type name
            edge_name = f"edge_index_{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
            obs_dict[edge_name] = edge_index.astype(np.int64)
    
    return obs_dict


def _build_observation_space_hetero_graph(sample_graph=None):
    """
    Build observation space for heterogeneous graphs compatible with gymnasium spaces.Dict
    
    Args:
        line_state: Line state object (optional)
        sample_graph: Sample HeteroData object to infer shapes
        
    Returns:
        spaces.Dict: Observation space compatible with gymnasium
    """
    
    # Create observation space based on HeteroData structure
    space_dict = {}
    
    # For each node type in the heterogeneous graph
    for node_type in sample_graph.node_types:
        if node_type in sample_graph.x_dict:
            node_features = sample_graph.x_dict[node_type]
            if isinstance(node_features, torch.Tensor):
                shape = tuple(node_features.shape)
            else:
                shape = tuple(np.array(node_features).shape)
            
            # Create Box space for node features
            space_dict[f"{node_type}_x"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32
            )
    
    # Add edge information (indices and attributes)
    for edge_type in sample_graph.edge_types:
        edge_type_str = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
        
        # Edge indices
        if edge_type in sample_graph.edge_index_dict:
            edge_index = sample_graph.edge_index_dict[edge_type]
            if isinstance(edge_index, torch.Tensor):
                shape = tuple(edge_index.shape)
            else:
                shape = tuple(np.array(edge_index).shape)
            
            space_dict[f"edge_index_{edge_type_str}"] = spaces.Box(
                low=0,
                high=np.iinfo(np.int64).max,  # Use proper max for int64
                shape=shape,
                dtype=np.int64
            )
        
        # Edge attributes
        if hasattr(sample_graph[edge_type], 'edge_attr') and sample_graph[edge_type].edge_attr is not None:
            edge_attr = sample_graph[edge_type].edge_attr
            if isinstance(edge_attr, torch.Tensor):
                shape = tuple(edge_attr.shape)
            else:
                shape = tuple(np.array(edge_attr).shape)
            
            space_dict[f"edge_attr_{edge_type_str}"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32
            )
    
    return spaces.Dict(space_dict)


class LineSimulation(gym.Env):
    """
    A Gym-compatible environment for simulating production lines using LineFlow.

    This environment wraps a LineFlow production line into a reinforcement learning setup, where
    agents can interact with stations via actions at discrete decision points, while the underlying
    process unfolds in continuous time using discrete-event simulation.

    Args:
        line (lineflow.simulation.line.Line): A LineFlow `Line` object representing the production
            line layout and behavior.
        simulation_end (int): The simulation end time (in simulation time units).
        reward (str, optional): The reward signal to use. Options are "parts" (default) for counting
            produced parts, or "uptime" for average utilization.
        part_reward_lookback (int, optional): Time window for computing average uptime-based rewards
            (used only if reward=`uptime`).
        render_mode (str or None, optional): Optional rendering mode. Currently supports "human" for
            visual rendering or None.
    """

    metadata = {"render_modes": [None, "human"]}

    def __init__(self, line, simulation_end, reward="parts", part_reward_lookback=0, render_mode=None):
        super().__init__()
        self.line = line
        self.simulation_end = simulation_end
        self.part_reward_lookback = part_reward_lookback

        self.render_mode = render_mode

        assert reward in ["uptime", "parts"]
        self.reward = reward
        self.use_graph_as_states = self.line.use_graph_as_states
        self.reward = reward
        
        self.action_space = _build_action_space(self.line.state)
        if self.use_graph_as_states:
            self.observation_space = _build_observation_space_hetero_graph(
                sample_graph=self.line._graph_states
            )
        else:
            self.observation_space = _build_observation_space(line_state=self.line.state)

        self.n_parts = 0
        self.n_scrap_parts = 0

    def _map_to_action_dict(self, actions):

        actions_iterator = filterfalse(
            lambda n: not self.line.state[n[0]][n[1]].is_actionable,
            self.line.state
        )

        actions_dict = {}
        for action, (station, action_name) in zip(actions, actions_iterator):
            if station not in actions_dict:
                actions_dict[station] = {}

            actions_dict[station][action_name] = action
        return actions_dict

    def step(self, actions):
        """
        Advances the simulation by one environment step.

        Args:
            actions (list or array): A list of agent actions corresponding to actionable features.

        Returns:
            observation (numpy.ndarray): Observation tensor representing the current state.
            reward (float): The computed reward for the current step.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode ended due to an internal error or simulation limit.
            info (dict): Additional diagnostic information.
        """
        actions = self._map_to_action_dict(actions)
        self.line.apply(actions)

        try:
            state, terminated = self.line.step(self.simulation_end)
            truncated = False
        except simpy.core.EmptySchedule:
            # TODO: not tested yet
            state = self.line.state
            terminated = True
            truncated = True

        if self.line.use_graph_as_states:
            observation = _convert_state_to_hetero_graph(state)
        else:
            observation = self._get_observations_as_tensor(state)
        # TODO: add work in process to the reward as penalty
        if self.reward == "parts":
            reward = (self.line.get_n_parts_produced() - self.n_parts) - \
                self.line.scrap_factor*(self.line.get_n_scrap_parts() - self.n_scrap_parts)
        elif self.reward == "uptime":
            reward = self.line.get_uptime(lookback=self.part_reward_lookback).mean()
        else:
            assert False, f"Reward {reward} not implemented"

        self.n_parts = self.line.get_n_parts_produced()
        self.n_scrap_parts = self.line.get_n_scrap_parts()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, self._get_info()
        # actions = self._map_to_action_dict(actions)
        # self.line.apply(actions)

        # try:
        #     state, terminated = self.line.step(self.simulation_end)
        #     truncated = False
        # except simpy.core.EmptySchedule:
        #     state = self.line.state
        #     terminated = True
        #     truncated = True

        # # Convert state to appropriate observation format
        # if self.use_graph_as_states:
        #     # Convert to HeteroData and then to dict for observation space compatibility
        #     hetero_graph = _convert_state_to_hetero_graph(state)
        #     observation = _convert_hetero_graph_to_dict(hetero_graph)
        # else:
        #     observation = self._get_observations_as_tensor(state)

        # # Calculate reward
        # if self.reward == "parts":
        #     reward = (self.line.get_n_parts_produced() - self.n_parts) - \
        #         self.line.scrap_factor*(self.line.get_n_scrap_parts() - self.n_scrap_parts)
        # elif self.reward == "uptime":
        #     reward = self.line.get_uptime(lookback=self.part_reward_lookback).mean()

        # self.n_parts = self.line.get_n_parts_produced()
        # self.n_scrap_parts = self.line.get_n_scrap_parts()

        # if self.render_mode == "human":
        #     self.render()

        # return observation, reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return self.line.info()

    def increase_scrap_factor(self, factor=0.1):
        """
        Sets the scrap penalty factor in the reward function.

        Args:
            factor (float): The multiplier applied to scrap parts in the parts-based reward.
        """
        self.line.scrap_factor = factor

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)

        self.line.reset(random_state=seed)
        self.n_parts = 0
        self.n_scrap_parts = 0

        state, _ = self.line.step()
        # observation vector as state
        if self.line.use_graph_as_states:
            # hetero_graph = _convert_state_to_hetero_graph(state)
            observation = _convert_hetero_graph_to_dict(state)
        else:
            observation = self._get_observations_as_tensor(state)

        if self.render_mode == "human":
            self.screen = self.line.setup_draw()
            self.render()
        return observation, self._get_info()

    @property
    def features(self):
        return self.line.state.observable_features

    def render(self):
        self.line._draw(self.screen)

    def close(self):
        if self.render_mode == 'human':
            self.line.teardown_draw()

    def _get_observations_as_tensor(self, state):

        X = state.get_observations(lookback=1, include_time=False)
        return np.array(X, dtype=np.float32)
    
    def _get_observations_as_dict(self, state):

        X = state.get_observations(lookback=1, include_time=False)
        return {k: np.array(v, dtype=np.float32) for k, v in X.items()}
