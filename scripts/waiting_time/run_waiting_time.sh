#!/bin/bash

INFO="[('S_component', 'waiting_time')]"
ENV="waiting_time"
TOTAL_STEPS=1000000
N=4

for SEED in 11 2024 174
do
    # ../run_ppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    /home/aipexws1/conan/agile_stem/lineflow/scripts/run_rppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    # ../run_a2c.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    # ../run_trpo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
done
