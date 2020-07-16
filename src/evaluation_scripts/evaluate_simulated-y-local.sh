#!/bin/bash

dir=$(pwd)/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# global settings
reps=$1
query_strategies="random ie-thresh ie-adj-cost alio ceal proactive mapal-1-0.0001-2-1-entropy"
data_sets="iris-simulated-y wine-simulated-y parkinsons-simulated-y prnn-craps-simulated-y sonar-simulated-y seeds-simulated-y glass-simulated-y
           vertebra-column-simulated-y ecoli-simulated-y ionosphere-simulated-y user-knowledge-simulated-y chscase-vine-simulated-y kc2-simulated-y breast-cancer-wisconsin-simulated-y
	         balance-scale-simulated-y blood-transfusion-simulated-y pima-indians-diabetes-simulated-y vehicle-simulated-y biodegradation-simulated-y banknote-simulated-y
	         steel-plates-fault-simulated-y segment-simulated-y phoneme-simulated-y satimage-simulated-y wind-simulated-y"
budget=0.4
test_ratio=0.4
results_path='results/simulated-y/csvs'

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

