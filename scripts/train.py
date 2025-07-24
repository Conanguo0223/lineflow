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
    MultiProcess,
    WorkerAssignment,
    ComplexLine,
    WaitingTime,
    WaterLine,
)
from sb3_contrib import (
    RecurrentPPO,
    TRPO,
)
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
)
from stable_baselines3 import (
    PPO,
    A2C,
)

from wandb.integration.sb3 import WandbCallback


def _make_line(name, n_cells, info, simulation_step_size=1, curriculum=False):

    if name == 'part_distribution':
        return MultiProcess(
            alternate=False,
            n_processes=n_cells,
            step_size=simulation_step_size,
            info=info,
        )

    if name == 'worker_assignment':
        return WorkerAssignment(
            n_assemblies=n_cells,
            with_rework=False,
            step_size=simulation_step_size,
            info=info,
        )

    if name == 'complex_line':
        return ComplexLine(
            alternate=False,
            n_assemblies=n_cells,
            n_workers=3*n_cells,
            scrap_factor=0 if curriculum else 1/n_cells,
            step_size=1,
            info=info,
            )

    if name == 'waiting_time':
        return WaitingTime(
            step_size=simulation_step_size,
            info=info,
            processing_time_source=5, 
        )

    if name == 'waiting_time_jump':
        return WaitingTime(
            step_size=simulation_step_size,
            info=info,
            processing_time_source=5, 
            with_jump=True, 
            t_jump_max=2000, 
            scrap_factor=1,
        )
    if name == 'water_line':
        return WaterLine(
            info=info,
            step_size=simulation_step_size,
            scrap_factor=1
            # use_graph_as_states=False,
            # scrap_factor=0 if curriculum else 1/n_cells,
        )
    raise ValueError('Unkown simulation')


def train(config):
    """
    Function that handles RL training

    Args:
    - `train`: Scores from the model update phase
    - `rollout`: Scores when a policy is rolled out to gather new experiences.
    - `eval`: Scores when a policy is evaluated on a separate environment

    Notes:
        Size of rollout-buffer: `n_steps*n_envs`, then an model-update is done
    """

    simulation_end = config['simulation_end'] + 1

    env_train = make_stacked_vec_env(
        line=_make_line(config['env'], config['n_cells'], config['info'], curriculum=config['curriculum']),
        simulation_end=simulation_end,
        reward=config["rollout_reward"],
        n_envs=config['n_envs'],
        n_stack=config['n_stack'] if not config['recurrent'] else 1,
    )

    env_eval = make_stacked_vec_env(
        line=_make_line(config['env'], config['n_cells'], config['info'], curriculum=config['curriculum']),
        simulation_end=simulation_end,
        reward=config["eval_reward"],
        n_envs=1,
        n_stack=config['n_stack'] if not config['recurrent'] else 1,
    )
    run = wandb.init(
        project='Lineflow',
        sync_tensorboard=True,
        config=config
    )
    log_path = os.path.join(config['log_dir'], run.id)

    if config['env'] == 'complex_line' and config['curriculum']:
        curriculum_callback = CurriculumLearningCallback(
            # Task marked as resolved if rewards is above 100
            threshold=100, 
            # Update of scrap factor
            update=(1/config["n_cells"])/5, 
            factor_max=1/config["n_cells"],
            look_back=3,
        )
    else:
        curriculum_callback = None

    eval_callback = EvalCallback(
        eval_env=env_eval,
        deterministic=config['deterministic'],
        n_eval_episodes=1,
        # Every (eval_freq*eval_envs) / (n_steps*train_envs)  step an update is done
        eval_freq=config["n_steps"]* config["n_envs"] * 10, # ever step in every env counts
        callback_after_eval=curriculum_callback,
    )

    model_args = {
        "policy": 'MlpPolicy' if not config['recurrent'] else 'MlpLstmPolicy',
        "env": env_train,
        "n_steps": config["n_steps"],
        "gamma": config['gamma'],  # discount factor
        "learning_rate": config["learning_rate"],
        "use_sde": False,
        "normalize_advantage": config['normalize_advantage'],
        "device": get_device(),
        "tensorboard_log": log_path,
        "stats_window_size": 10,
        "verbose": 0,
        "seed": config['seed'] if config['seed'] != 0 else None,
    }

    if "PPO" in config['model']:
        model_cls = PPO
        model_args["batch_size"] = config["n_steps"]  # mini-batch size
        model_args["n_epochs"] = 5  # number of times to go over experiences with mini-batches
        model_args["clip_range"] = config['clip_range']
        model_args["max_grad_norm"] = 0.5
        model_args["ent_coef"] = config['ent_coef']

        if config['recurrent']:
            model_cls = RecurrentPPO

    if "A2C" in config['model']:
        model_cls = A2C
        model_args["max_grad_norm"] = 0.5
        model_args["ent_coef"] = config['ent_coef']

    if "TRPO" in config['model']:
        model_cls = TRPO

    model = model_cls(**model_args)

    model.learn(
        total_timesteps=config["total_steps"],
        callback=CallbackList([
            WandbCallback(verbose=0),
            eval_callback,
        ])
    )
    run.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="worker_assignment", type=str)
    parser.add_argument('--n_cells', default=3, type=int)
    parser.add_argument('--model', default="PPO", type=str)
    parser.add_argument('--learning_rate', default=0.0003, type=float)
    parser.add_argument('--ent_coef', default=0.1, type=float)
    parser.add_argument('--n_stack', default=1, type=int)
    parser.add_argument('--n_steps', default=500, type=int) # Tim until update is done
    parser.add_argument('--n_envs', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--total_steps', default=500_000, type=int)
    parser.add_argument(
        '--log_dir', 
        default="./logs",
        type=str,
        help="Location where tensorboard logs are saved",
    )

    parser.add_argument(
        '--simulation_end',
        default=4000,
        type=int,
        help="time of simulation, an update is done after every simulation",
    )
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--clip_range', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--normalize_advantage', action='store_true', default=False)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--curriculum', action='store_true', default=False)

    parser.add_argument(
        '--info',
        type=str,
        default="[]",
        help="Station info that should be logged like \"[('A1', 'waiting_time')]\""
    )

    parser.add_argument('--eval_reward', default="parts", type=str)
    parser.add_argument('--rollout_reward', default="parts", type=str)

    config = parser.parse_args().__dict__
    config['info'] = ast.literal_eval(config['info'])

    train(config)
