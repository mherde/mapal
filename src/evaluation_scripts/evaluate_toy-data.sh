#!/bin/bash

dir=$(pwd)/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# global settings
reps=$1
query_strategies="mapal-1-0.0001-1-1-entropy mapal-1-0.0001-2-1-entropy mapal-1-0.0001-3-1-entropy mapal-1-1.0-2-1-entropy mapal-1-0.01-2-1-entropy mapal-1-0.0001-2-1-entropy"
data_sets="large-example-data-set-x"
budget=200
test_ratio=0.8
results_path='results/simulated-x/csvs'

# execute each configuration
i=0
for d in ${data_sets}; do
  for q in ${query_strategies}; do
    for s in {1..100}; do
      i=$(("$i"+1))
      python -u "$dir"/evaluation_scripts/experimental_setup.py --data_set "$d" --results_path "$results_path" --test_ratio "$test_ratio" --query_strategy "$q" --budget "$budget" --seed "$s" &
      if [ $(( s % $reps )) -eq 0 ]; then
        wait
      else
        printf "Continue"
      fi
    done
    wait
  done
done
echo "$i"
wait

