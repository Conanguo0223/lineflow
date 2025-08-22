import sys
import numpy as np
import torch
from itertools import (
    cycle,
    zip_longest,
)


def zip_cycle(*lengths):
    iterables = [range(n) for n in lengths]
    cycles = [cycle(i) for i in iterables]
    for _ in zip_longest(*iterables):
        yield tuple(next(i) for i in cycles)


def compute_processing_times_of_parts(line, station_name, finished_only=False):
    df = line.get_observations(station_name)
    df_working = df[df['mode'] == 'working']
    df_times = df_working.groupby('carrier').apply(
        lambda x: x['T_end'].max() - x['T_start'].min(), include_groups=False
    ).to_frame('time')

    if finished_only:
        # For finished parts, the last state must be waiting
        df_last_modes = df.groupby('carrier').apply(lambda df_: df_['mode'].values[-1])

        finished_parts = df_last_modes[df_last_modes == 'waiting'].index
        df_times = df_times[df_times.index.isin(finished_parts)]
    return df_times


def get_device():
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        return "mps"

    if sys.platform == "linux" and torch.cuda.is_available():
        return torch.device("cuda:0")

    return "cpu"


def switch_states_actionablility(line, lineobjects, make_actionable):
    """
    Switch the actionability of the states of lineobjects.

    Args:
        line (lineflow.simulation.lines.line): The line object
        lineobjects (list): A list of line object names for which the actionablility of states is
            switched
        make_actionable (bool): Weather to make states actionable or not
    """
    for object in lineobjects:
        for state in line._objects[object].state.states.keys():
            line._objects[object].state.states[state].is_actionable = make_actionable


def torch_to_numpy(samples):

    for o_name in samples.keys():
        for a_name in samples[o_name].keys():
            samples[o_name][a_name] = samples[o_name][a_name].cpu().numpy()
    return samples


def compute_performance_coefficient(n, c=0.3):
    """
    Compute the performance of a cell for increased workers
    """
    return np.exp(-c*(n-1))

# ============== helpers for GNN visualizations ==============

