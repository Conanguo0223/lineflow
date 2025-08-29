#!/bin/bash

echo "Start PPO"

SEED=$1
ENV=$2
N=$3
INFO=$4
TOTAL_STEPS=$5
LR=("0.01" "0.001" "0.0001" "0.01" "0.001" "0.0001")
STACK=("40" "40" "40" "100" "100" "100")

for i in "${!LR[@]}"; do
    echo "Running config $i: lr=${LR[$i]}, stack=${STACK[$i]}"
    python3 train.py --deterministic \
        --env="$ENV" --n_cells="$N" --model=PPO --curriculum \
        --seed="$SEED" --info="$INFO" --total_step="$TOTAL_STEPS" \
        --learning_rate="${LR[$i]}" \
        --n_stack="${STACK[$i]}" \
done
