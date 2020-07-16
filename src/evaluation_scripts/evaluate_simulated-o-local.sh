#!/bin/bash

dir=$(pwd)/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# global settings
reps=$1
query_strategies="random ie-thresh ie-adj-cost alio ceal proactive mapal-1-0.0001-2-1-entropy"
data_sets="iris-simulated-o wine-simulated-o parkinsons-simulated-o prnn-craps-simulated-o sonar-simulated-o seeds-simulated-o glass-simulated-o
           vertebra-column-simulated-o ecoli-simulated-o ionosphere-simulated-o user-knowledge-simulated-o chscase-vine-simulated-o kc2-simulated-o breast-cancer-wisconsin-simulated-o
	         balance-scale-simulated-o blood-transfusion-simulated-o pima-indians-diabetes-simulated-o vehicle-simulated-o biodegradation-simulated-o banknote-simulated-o
	         steel-plates-fault-simulated-o segment-simulated-o phoneme-simulated-o satimage-simulated-o wind-simulated-o"
budget=0.4
test_ratio=0.4
results_path='results/simulated-o/csvs'

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

