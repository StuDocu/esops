#!/bin/bash

# Parameters
NUM_RUNS=$1
SEED=$2
NUM_ITEMS=$3
LOW=$4
HIGH=$5
TIME_SPAN=$6
ALPHA=$7
GAMMA=$8
EPSILON=$9

# Check if all parameters are provided
if [ "$#" -ne 9 ]; then
    echo "Usage: $0 <num_runs> <seed> <num_items> <low> <high> <time_span> <alpha> <gamma> <epsilon>"
    exit 1
fi

# Run the simulation N times
for ((i=0; i<$NUM_RUNS; i++))
do
    # Run the simulation command with the provided parameters
    poetry run simulate --seed $SEED --name n=$NUM_ITEMS.low=$LOW.high=$HIGH.ts=$TIME_SPAN.alpha=$ALPHA.gamma=$GAMMA run-q-learning --alpha $ALPHA --gamma $GAMMA --epsilon $EPSILON --num-items $NUM_ITEMS --time-span $TIME_SPAN --low $LOW --high $HIGH

    echo "Run $((i+1)) completed."
done

echo "All $NUM_RUNS runs completed."
