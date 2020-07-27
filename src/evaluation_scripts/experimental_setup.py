import os

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import numpy as np
import pandas as pd
import math

from argparse import ArgumentParser

from functools import partial

from annotlib.standard import StandardAnnot
from src.base.data_set import DataSet
from src.models.parzen_window_classifier import PWC
from src.models.beta_annotators_model import BAM
from src.query_strategies.cost_effective_active_learning import CEAL
from src.query_strategies.active_learning_with_imperfect_oracles import ALIO
from src.query_strategies.interval_estimate_threshold import IEThresh
from src.query_strategies.interval_estimate_adjusted_cost import IEAdjCost
from src.query_strategies.proactive_learning import Proactive
from src.query_strategies.multi_annotator_probabilistic_active_learning import MAPAL
from src.query_strategies.random_sampling import RS
from src.utils.data_functions import load_data, investigate_data_set
from src.utils.mathematical_functions import estimate_bandwidth
from src.utils.statistic_functions import eval_perfs, eval_annot_stats, misclassification_costs

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_kernels

from time import time


def run(results_path, data_set, query_strategy, budget, test_ratio, seed):
    """
    Run experiments to compare query selection strategies.
    Experimental results are stored in a .csv-file.

    Parameters
    ----------
    results_path: str
        Absolute path to store results.
    data_set: str
        Name of the data set.
    query_strategy: str
        Determines query strategy.
    budget: int
        Maximal number of labeled samples.
    test_ratio: float in (0, 1)
        Ratio of test samples.
    seed: float
        Random seed.
    """
    # --------------------------------------------- LOAD DATA ----------------------------------------------------------
    is_cosine = 'reports' in data_set
    X, y_true, y = load_data(data_set_name=data_set)
    n_features = np.size(X, axis=1)
    n_classes = len(np.unique(y))
    n_annotators = np.size(y, axis=1)
    print(data_set + ': ' + str(investigate_data_set(data_set)))
    budget_str = str(budget)
    if budget > len(X) * n_annotators * (1 - test_ratio):
        budget = int(math.floor(len(X) * n_annotators * (1 - test_ratio)))
    elif budget > 1:
        budget = int(budget)
    elif 0 < budget <= 1:
        budget = int(math.floor(len(X) * n_annotators * (1 - test_ratio) * budget))
    else:
        raise ValueError("'budget' must be a float in (0, 1] or an integer in [0, n_samples]")
    budget = np.min((budget, 1000))

    # --------------------------------------------- STATISTICS ---------------------------------------------------------
    # define storage for performances
    results = {}

    # define performance functions
    C = 1 - np.eye(n_classes)

    perf_funcs = {'micro-misclf-rate': [partial(misclassification_costs, C=C, average='micro'), {}],
                  'macro-misclf-rate': [partial(misclassification_costs, C=C, average='macro'), {}]}

    # ------------------------------------------- LOAD DATA ----------------------------------------------------
    print('seed: {}'.format(str(seed)))
    X_train, X_test, y_true_train, y_true_test, y_train, y_test = train_test_split(X, y_true, y, test_size=test_ratio,
                                                                                   random_state=seed)
    while not np.array_equal(np.unique(y_true_train), np.unique(y_true_test)):
        X_train, X_test, y_true_train, y_true_test, y_train, y_test = train_test_split(X, y_true, y, random_state=seed,
                                                                                       test_size=test_ratio)
        seed += 1000
        print('new seed: {}'.format(seed))
    n_samples = len(X_train)

    # --------------------------------------------- CSV NAMES ----------------------------------------------------------
    csv_name = '{}_{}_{}_{}_{}.csv'.format(data_set, query_strategy, budget_str, test_ratio, seed)

    # ------------------------------------------ PREPROCESS DATA -------------------------------------------------------
    # standardize data
    if is_cosine:
        kwargs = {'metric': 'cosine'}
    else:
        # standardize data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # compute bandwidth
        bandwidth = estimate_bandwidth(n_samples=n_samples, n_features=n_features)
        print('bandwidth: {}'.format(str(bandwidth)))
        gamma = 0.5 * (bandwidth ** (-2))

        kwargs = {'metric': 'rbf', 'gamma': gamma}

    # setup classifiers
    pwc_train = PWC(n_classes=n_classes, combine_labels=False, random_state=seed, **kwargs)
    S_train = pairwise_kernels(X_train, X_train, **kwargs)
    pwc_test = PWC(n_classes=n_classes, metric='precomputed', combine_labels=False, probabilistic=False,
                   random_state=seed)
    S_test = pairwise_kernels(X_test, X_train, **kwargs)

    # set up data set
    data_set = DataSet(X_train, n_annotators=n_annotators)
    annotators = StandardAnnot(X=X_train, Y=y_train)

    # create query strategy
    if query_strategy == 'ceal':
        query_strategy = CEAL(data_set=data_set, n_classes=n_classes, clf=pwc_train, n_neighbors=10,
                              label_proportion=0.2 * budget / n_annotators, random_state=seed, **kwargs)
    elif query_strategy == 'alio':
        query_strategy = ALIO(data_set=data_set, n_classes=n_classes, clf=pwc_train,
                              label_proportion=0.2 * budget / n_annotators, random_state=seed)
    elif query_strategy == 'proactive':
        query_strategy = Proactive(data_set=data_set, n_classes=n_classes, clf=pwc_train, n_components=20,
                                   label_proportion=0.2 * budget / n_annotators, random_state=seed)
    elif 'mapal' in query_strategy:
        params = query_strategy.split('-')
        mean_prior = float(params[1])
        sum_prior = (np.sum(S_train) - n_samples) / (n_samples ** 2 - n_samples) if params[2] == 'mean' else float(
            params[2])
        prior = np.array([mean_prior, 1 - mean_prior])
        prior /= np.sum(prior)
        prior *= sum_prior
        print('prior = {}'.format(prior))
        m_max = int(params[3])
        alpha = float(params[4])
        weights_type = str(params[5])
        bam = BAM(n_classes=n_classes, weights_type=weights_type, prior=prior, random_state=seed, **kwargs)
        query_strategy = MAPAL(data_set=data_set, m_max=m_max, n_classes=n_classes, S=S_train, bam=bam,
                               alpha_x=alpha, alpha_c=alpha, random_state=seed)
    elif query_strategy == 'ie-adj-cost':
        query_strategy = IEAdjCost(data_set=data_set, clf=pwc_train, n_classes=n_classes, delta=0.4, lmbda=0.4,
                                   alpha=0.05, epsilon=0.8, random_state=seed)
    elif query_strategy == 'ie-thresh':
        query_strategy = IEThresh(data_set=data_set, clf=pwc_train, n_classes=n_classes, epsilon=0.8, alpha=0.05,
                                  random_state=seed)
    elif query_strategy == 'random':
        query_strategy = RS(data_set=data_set, random_state=seed)
    else:
        raise ValueError(
            "query strategy must be in ['ceal', 'ie-thresh', 'pal-1-all', 'pal-1-single', 'mapal-..., random]")

    # ----------------------------------------- ACTIVE LEARNING CYCLE --------------------------------------------------
    times = [0]
    for b in range(budget):
        print("budget: {}".format(b))
        # evaluate results
        eval_perfs(clf=pwc_test, X_train=S_train, y_train=y_true_train, X_test=S_test, y_test=y_true_test,
                   perf_results=results, perf_funcs=perf_funcs)
        eval_annot_stats(y=data_set.y_, y_true=y_true_train, results=results)

        # select sample and annotator
        t = time()
        selection = query_strategy.make_query()
        times.append(time() - t)
        sample_id = selection[0, 0]
        annotator_id = [selection[0, 1]]
        print("selected sample: {}".format(sample_id))
        print("selected annotator: {}".format(annotator_id))

        # query selected annotator for labeling selected sample
        X_query = [X_train[sample_id]]
        y_query = annotators.class_labels(X_query, annotator_ids=annotator_id)
        print('class label: {}'.format(y_query[0, annotator_id[0]]))

        # update training data
        data_set.update_entries(sample_id, y_query)
        print(data_set.len_labeled(per_annotator=True))

        # retrain classifier
        pwc_test.fit(X=data_set.X_, y=data_set.y_, c=data_set.c_)

    # evaluate results
    eval_perfs(clf=pwc_test, X_train=S_train, y_train=y_true_train, X_test=S_test, y_test=y_true_test,
               perf_results=results, perf_funcs=perf_funcs)
    eval_annot_stats(y=data_set.y_, y_true=y_true_train, results=results)

    # store performance results
    results['times'] = times
    df = pd.DataFrame(results)
    df.to_csv('{}/{}'.format(results_path, csv_name), index_label='index')


def main():
    parser = ArgumentParser(description='Parameters of experimental setup')
    parser.add_argument('--results_path', type=str, help='absolute path for storing results in form of .csv file')
    parser.add_argument('--data_set', type=str, help='name of data set')
    parser.add_argument('--query_strategy', default='random', type=str, help='name of active learning strategy')
    parser.add_argument('--budget', type=float, default=0.4, help='percentage of annotation acquisitions')
    parser.add_argument('--test_ratio', type=float, default=0.4, help='ratio of test samples')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    args = parser.parse_args()
    run(results_path=args.results_path, data_set=args.data_set, query_strategy=args.query_strategy, budget=args.budget,
        test_ratio=args.test_ratio, seed=args.seed)


if __name__ == '__main__':
    main()
