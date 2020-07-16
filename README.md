# Multi-annotator Probabilistic Active Learning

Authors: Marek Herde, Daniel Kottke, Denis Huseljic, and Bernhard Sick

## Project Structure
- data: contains .csv-files of data sets being not available at [OpenML](https://www.openml.org/home)
- plots: directory where the visualizations of MaPAL will be saved
- results: path where all results will be stored including csvs, learning curves, and ranking statistics
- src: Python package consisting of several sub-packages
    - base: implementation of DataSet and QueryStrategy class
    - classifier: implementation of Similarity based Classifier (SbC) being an advancement of the Parzen Window Classifier (PWC) 
    - evaluation_scripts: scripts for experimental setup
    - notebooks: jupyter notebooks for the investigation of MaPAL, simulation of annotators, and the illustration of results
    - query_strategies: implementation of all query/AL strategies
    - utils: helper functions

## How to execute an experiment?
Due to the large number of experiments, we executed the experiments on a computer cluster. Using these nodes, we were able to execute 100 experiments simultaneously. 

Without such a computer cluster, it will  probably take several days to reproduce all results of the article. Nevertheless, one can execute the 
experiments on a local personal computer by following the upcoming steps.

1. Setup Python environment:
```bash
projectpath$ sudo apt-get install python3-pip
projectpath$ pip3 install virtualenv
projectpath$ virtualenv mapal
projectpath$ source mapal/bin/activate
projectpath$ pip3 install -r requirements.txt
```
2. Simulate annotators: Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/notebooks/simulate_annotators.ipynb`. This must be the first step before executing any experiment.
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```
2. Get information about the available hyperparameters (argparse) for the experiments.
```bash
projectpath$ source mapal/bin/activate
projectpath$ export PYTHONPATH="${PYTHONPATH}":$(pwd)/src
projectpath$ python3 src/evaluation_scripts/experimental_setup.py -h
```
3. Example experiment: To test MaPAL with M_max=2 and beta_0=0.0001 on the dataset iris with annotators having instance-dependent performance values and with
    - a budget of 40% of all available annotations, 
    - a test ratio of 40%, 
    - and using the seed 1,
    
we have to execute the following commands:
```bash
projectpath$ source mapal/bin/activate
projectpath$ export PYTHONPATH="${PYTHONPATH}":$(pwd)/src
projectpath$ python3 src/evaluation_scripts/experimental_setup.py \
  --query_strategy mapal-1-0.0001-2-1-entropy \
  --data_set iris-simulated-x \
  --results_path results/simulated-x/csvs \
  --test_ratio 0.4 \
  --budget 0.4 \
  --seed 1
```
For this example, the results are saved in the directory `projectpath/results/simulated-x/csvs/` as a .csv-file.

The names of the possible data sets are given in the following files:
- `projectpath/data/data-set-names-real-world.csv`: contains the names of the data sets with real-world annotators (the data set grid is not available because it contains confidential data),
- `projectpath/data/data-set-names-simulated-o.csv`: contains the names of the data sets with simulated annotators having uniform performance values,
- `projectpath/data/data-set-names-simulated-y.csv`: contains the names of the data sets with simulated annotators having class-dependent performance values,
- `projectpath/data/data-set-names-simulated-x.csv`: contains the names of the data sets with simulated annotators having instance-dependent performance values.

To create the ranking statistics, there must be at least one run for each strategy on a data set.  The different AL strategies that can be used as `--query_strategy` argument are given in the following:
- MaPAL: `mapal-1-0.0001-2-1-entropy`,
- IEThresh: `ie-thresh`,
- IEAdjCost: `ie-adj-cost`,
- CEAL: `ceal`,
- ALIO: `alio`,
- Proactive: `proactive`,
- Random: `random`.

To conduct the experiments data sets with real-world annotators in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation/evaluate_real-world-local.sh 5
```
The argument `5` is an example and gives the maximum number of runs that can be executed in parallel. You can change this number.

To conduct the experiments data sets with simulated annotators having uniform performances values in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_simulated-o-local.sh 5
```

To conduct the experiments data sets with simulated annotators having class-dependent performances values in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_simulated-y-local.sh 5
```

To conduct the experiments data sets with simulated annotators having instance-dependent performances values in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_simulated-x-local.sh 5
```

## How to illustrate the experimental results?
Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/notebooks/experimental_results.ipynb`.
Remark: The ranking plots can only be created when we have for each dataset and each strategy the same number of 
executed experiments. 
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```

## How to reproduce the annotation performance and instance utility plots?
Start jupyter-notebook and run the jupyter-notebook file `mapal/src/notebooks/visualization.ipynb`.
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```
