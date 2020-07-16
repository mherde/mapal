#!/bin/bash

dir=$(pwd)/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# global settings
# shellcheck disable=SC1068
reps=$1
query_strategies="random ie-thresh ie-adj-cost alio ceal proactive mapal-1-0.0001-2-1-entropy"
data_sets="medical reports-mozilla reports-compendium"
budget=0.4
test_ratio=0.4
results_path='results/real-world/csvs'

# execute each configuration
for d in ${data_sets}; do
  for q in ${query_strategies}; do
    for s in {1..100}; do
      echo $s
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
wait

