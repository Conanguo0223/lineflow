#!/bin/bash

echo "Start PPO"

SEED=$1
ENV=$2
N=$3
INFO=$4
TOTAL_STEPS=$5
SIM_STEPS=$6
REWARD_TYPE=$7
USE_NORMALIZED=$8

LR=("0.001" "0.0001" "0.001" "0.0001" )
ENTCOEF=( "0.01"    "0.01" )

for i in "${!LR[@]}"; do
    echo "Hello Running config $i: lr=${LR[$i]}, ent_coef=${ENTCOEF[$i]}, sim_steps=${SIM_STEPS}, reward_type=${REWARD_TYPE}, use_normalized=${USE_NORMALIZED}"
    python3 /home/aipexws1/conan/agile_stem/lineflow/scripts/train.py --deterministic \
        --env="$ENV" --n_cells="$N" --model=PPO --n_stack=1 --recurrent \
        --seed="$SEED" --info="$INFO" --total_step="$TOTAL_STEPS" \
        --learning_rate="${LR[$i]}" \
        --ent_coef="${ENTCOEF[$i]}" \
        --eval_reward="parts" \
        --rollout_reward="$REWARD_TYPE" \
        --simulation_step_size="$SIM_STEPS" \
        --use_normalized="$USE_NORMALIZED"
done
