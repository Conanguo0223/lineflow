"""
This file is a wrapper around our simulation such that it is compatible with the stable-baselines
repo
"""
from itertools import filterfalse
import simpy
import gymnasium as gym
import numpy as np
from gymnasium import spaces

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

    def __init__(self, line, simulation_end, reward="parts", part_reward_lookback=0, render_mode=None, use_graph = False):
        super().__init__()
        self.line = line
        self.simulation_end = simulation_end
        self.part_reward_lookback = part_reward_lookback

        self.render_mode = render_mode

        assert reward in ["uptime", "parts"]
        self.reward = reward
        self.use_graph = use_graph
        if self.use_graph :
            # action stays the same, but observation space is different because we want to pass node features
            self.action_space = _build_action_space(self.line.state)
            self.observation_space = _build_observation_space_graph(line_state=self.line.state)
        else:
            # fix an order of states
            self.action_space = _build_action_space(self.line.state)
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
            observation = state
        else:
            observation = self._get_observations_as_tensor(state)

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
            observation = state
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
