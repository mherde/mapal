#!/bin/bash

dir=$(pwd)/src
export PYTHONPATH="${PYTHONPATH}":"$dir"

# global settings
reps=$1
query_strategies="random ie-thresh ie-adj-cost alio ceal proactive mapal-1-0.0001-2-1-entropy"
data_sets="iris-simulated-x wine-simulated-x parkinsons-simulated-x prnn-craps-simulated-x sonar-simulated-x seeds-simulated-x glass-simulated-x
           vertebra-column-simulated-x ecoli-simulated-x ionosphere-simulated-x user-knowledge-simulated-x chscase-vine-simulated-x kc2-simulated-x breast-cancer-wisconsin-simulated-x
	         balance-scale-simulated-x blood-transfusion-simulated-x pima-indians-diabetes-simulated-x vehicle-simulated-x biodegradation-simulated-x banknote-simulated-x
	         steel-plates-fault-simulated-x segment-simulated-x phoneme-simulated-x satimage-simulated-x wind-simulated-x"
budget=0.4
test_ratio=0.4
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

