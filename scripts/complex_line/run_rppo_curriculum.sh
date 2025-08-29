#!/bin/bash

echo "Start PPO"

SEED=$1
ENV=$2
N=$3
INFO=$4
TOTAL_STEPS=$5
SIM_STEPS=$6

declare -a LR=("0.0001" "0.0005" "0.001")

for i in "${!LR[@]}"; do
    echo "Running config $i: lr=${LR[$i]}"
    python3 train.py --deterministic \
        --env="$ENV" --n_cells="$N" --model=PPO --n_stack=1 --recurrent --curriculum \
        --seed="$SEED" --info="$INFO" --total_step="$TOTAL_STEPS" \
        --learning_rate="${LR[$i]}" \
        --ent_coef=0 \
        --simulation_end="$SIM_STEPS"
done
