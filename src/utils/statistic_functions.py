import numpy as np
import os
import glob
import pandas as pd

from scipy.stats import rankdata
from scipy.stats import wilcoxon

from sklearn.metrics import confusion_matrix
from sklearn.utils import column_or_1d, check_consistent_length, check_array


def misclassification_costs(y_true, y_pred, C, average='micro'):
    """
    Computes mean misclassification costs.

    Parameters
    ----------
    y_pred: array-like, shape (n_labels)
        Predicted class labels.
    y_true: array-like, shape (n_labels)
        True class labels.
    C: cost matrix, shape (n_classes, n_classes)
        Cost matrix with entry C[x, y] describing the costs of predicting class x when class y is true.
    average: 'micro'|'macro'
        Type of average.

    Returns
    -------
    mc: float
        Misclassifcation costs.
    """
    # check parameters
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)
    C = check_array(C)
    n_classes = C.shape[0]

    # compute mean misclassification costs
    if average == 'micro':
        return np.mean(C[y_true, y_pred])
    elif average == 'macro':
        labels = np.arange(n_classes)
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        samples_per_class = np.sum(conf_matrix, axis=1)
        mc_per_class = [np.sum([C[y, y_] * conf_matrix[y, y_] for y_ in labels]) / samples_per_class[y] for y in
                        labels]
        return np.nanmean(mc_per_class)
    else:
        raise ValueError("Parameter 'average' must be either 'micro' or 'macro'.")


def eval_perfs(clf, X_train, y_train, X_test, y_test, perf_funcs, perf_results=None):
    """
    Evaluates performances of a classifier on train and test data given a list of performance functions.
    Stores evaluated performances into a given dictionary.

    Parameters
    ----------
    clf: model
        Model to be evaluated. Must implement predict method.
    X_train: array-like, shape (n_training_samples, n_features)
        Training samples.
    y_train: array-like, shape (n_training_samples)
        Class labels of training samples.
    X_train: array-like, shape (n_test_samples, n_features)
        Test samples.
    y_train: array-like, shape (n_test_samples)
        Class labels of test samples.
    perf_funcs: dict-like
        Dictionary of performance functions to be used where 'y_true' and 'y_pred' are expected as parameters.
        An example entry is given by perf_funcs['key'] = [perf_func, kwargs], where 'kwargs' are keyword-only arguments
        passed to the 'predict' method of 'clf'.
    perf_results: dict-like, optional (default={})
        Dictionary of performances.

    Returns
    -------
    perf_results: dict-like
        Dictionary of updated performances.
    """
    # check parameters
    if not callable((getattr(clf, 'predict', None))):
        raise TypeError("'clf' must be an instance with the method 'predict'")
    perf_results = {} if perf_results is None else perf_results
    if not isinstance(perf_results, dict):
        raise TypeError("'perf_results' must be a dictionary")
    if not isinstance(perf_funcs, dict):
        raise TypeError("'perf_funcs' must be a dictionary")

    # create storage for performance measurements
    if len(perf_results) == 0:
        for key in perf_funcs:
            perf_results['train-' + key] = []
            perf_results['test-' + key] = []

    # compute performances
    for key, item in perf_funcs.items():
        perf_func = item[0]
        kwargs = item[1]
        y_train_pred = clf.predict(X_train, **kwargs)
        y_test_pred = clf.predict(X_test, **kwargs)
        perf_results['train-' + key].append(perf_func(y_pred=y_train_pred, y_true=y_train))
        perf_results['test-' + key].append(perf_func(y_pred=y_test_pred, y_true=y_test))

    return perf_results


def eval_annot_stats(y, y_true, results=None):
    """
    Evaluates several statistics regarding the labels provided by the annotators.

    Parameters
    ----------
    y: array-like, shape (n_samples, n_annotators)
        Class labels provided by the annotators.
    y_true: array-like, shape (n_test_samples)
        True class labels.
    results: dict-like, optional (default={})
        Dictionary for storing annotation statistics.

    Returns
    -------
    results: dict-like
        Dictionary for storing annotation statistics.
    """
    # check parameters
    y = check_array(y, force_all_finite=False)
    y_true = column_or_1d(y_true)
    check_consistent_length(y, y_true)
    n_annotators = y.shape[1]
    results = {} if results is None else results
    if not isinstance(results, dict):
        raise TypeError("'results' must be a dictionary")

    # determine labeled entries
    is_labeled = ~np.isnan(y)
    n_labeled_samples = np.sum(np.sum(is_labeled, axis=1) > 0)
    if 'n-labeled-samples' not in results:
        results['n-labeled-samples'] = []
    results['n-labeled-samples'].append(n_labeled_samples)

    # compute performances
    n_false_labels = 0
    n_true_labels = 0
    for a in range(n_annotators):
        is_correct = np.equal(y_true[is_labeled[:, a]], y[is_labeled[:, a], a])
        n_true_labels_a = np.sum(is_correct)
        n_false_labels_a = len(is_correct) - n_true_labels_a
        if 'n-true-labels-{}'.format(a) not in results:
            results['n-true-labels-{}'.format(a)] = []
        results['n-true-labels-{}'.format(a)].append(n_true_labels_a)
        if 'n-false-labels-{}'.format(a) not in results:
            results['n-false-labels-{}'.format(a)] = []
        results['n-false-labels-{}'.format(a)].append(n_false_labels_a)
        n_false_labels += n_false_labels_a
        n_true_labels += n_true_labels_a

    if 'n-true-labels' not in results:
        results['n-true-labels'] = []
    results['n-true-labels'].append(n_true_labels)
    if 'n-false-labels' not in results:
        results['n-false-labels'] = []
    results['n-false-labels'].append(n_false_labels)

    return results


def compute_statistics(dic):
    """
    Calculation of means and standard deviations of the lists stored in a given dictionary.

    Parameters
    ----------
    dic: dictionary, shape = {key_1: [[...]], key_n: [[...]]}
        Dictionary with lists of lists as values.

    Returns
    -------
    statistics: dictionary, shape = {key-1-mean: [...], key-1-std: [...], ..., key-n-mean: [...], key-n-std: [...]}
        Dictionary with lists of std and mean values.
    """
    if not isinstance(dic, dict):
        raise TypeError("'dic' must be a dictionary")

    statistics = {}

    # Iteration over all keys.
    for key in dic:
        # Calculation of means and std values.
        arr = np.array(dic[key])
        if arr.ndim == 2 and np.size(arr, axis=0) > 1:
            statistics[key + '-mean'] = np.mean(arr, axis=0)
            statistics[key + '-std'] = np.std(arr, ddof=1, axis=0)
        elif arr.ndim == 2 and np.size(arr, axis=0) == 1:
            statistics[key + '-mean'] = dic[key][0]
            statistics[key + '-std'] = len(dic[key][0]) * 0
        elif arr.ndim == 1:
            statistics[key + '-mean'] = np.mean(dic[key])
            statistics[key + '-std'] = np.std(dic[key])

    return statistics


def read_results(abs_path, data_set_names, strategy_names, budget, test_ratio, n_annotators=0):
    results = {}
    for d_idx, d in enumerate(data_set_names):
        results[d] = {}
        d_is_available = False
        for q_idx, q in enumerate(strategy_names):
            results[d][q] = {}
            results[d][q]['seed-list'] = []
            results[d][q]['test-misclf-rate'] = []
            results[d][q]['test-aulc'] = []
            results[d][q]['train-misclf-rate'] = []
            results[d][q]['train-aulc'] = []
            results[d][q]['n-labeled-samples'] = []
            results[d][q]['n-false-labels'] = []
            results[d][q]['times'] = []
            for a in range(n_annotators):
                results[d][q]['n-false-labels-{}'.format(a)] = []
                results[d][q]['n-true-labels-{}'.format(a)] = []
            csv_name = '{}_{}_{}_{}'.format(d, q, budget, test_ratio)
            listing = glob.glob(os.path.join(abs_path, '{}*'.format(csv_name)))
            for file in listing:
                d_is_available = True
                df = pd.read_csv(file)
                seed = int(file.split('.')[-2].split('_')[-1])
                results[d][q]['seed-list'].append(seed)
                results[d][q]['test-misclf-rate'].append(df['test-micro-misclf-rate'])
                results[d][q]['test-aulc'].append(np.mean(df['test-micro-misclf-rate']))
                results[d][q]['train-misclf-rate'].append(df['train-micro-misclf-rate'])
                results[d][q]['train-aulc'].append(np.mean(df['train-micro-misclf-rate']))
                results[d][q]['n-labeled-samples'].append(df['n-labeled-samples'])
                results[d][q]['n-false-labels'].append(df['n-false-labels'])
                results[d][q]['times'].append(np.mean(df['times']))
                for a in range(n_annotators):
                    results[d][q]['n-false-labels-{}'.format(a)].append(df['n-false-labels-{}'.format(a)])
                    results[d][q]['n-true-labels-{}'.format(a)].append(df['n-true-labels-{}'.format(a)])
            if results[d][q]['seed-list']:
                sort_indices = np.array(np.argsort(results[d][q]['seed-list']), dtype=int)
                for key in results[d][q]:
                    results[d][q][key] = np.array(results[d][q][key])[sort_indices]
            else:
                results[d].pop(q)
        if not d_is_available:
            results.pop(d)

    return results



def extract_execution_times(dic):
    execution_times_dic = {}
    for d_idx, d in enumerate(dic.keys()):
        execution_times_dic[d] = {}
        for q_idx, q in enumerate(dic[d].keys()):
            execution_times_dic[d][q] = np.mean(dic[d][q]['times'])
    return execution_times_dic
    

def compute_aulc_ranks(dic, c, eval_type='test'):
    aulc_ranks = None
    aulc_vals = []
    for d_idx, d in enumerate(dic.keys()):
        aulc_vals_d = []

        for q_idx, q in enumerate(dic[d].keys()):
            if aulc_ranks is None:
                aulc_ranks = np.zeros((len(dic[d]), len(dic)))
                test_results = np.zeros((len(dic[d]), len(dic)))
            aulc_vals_d.append(dic[d][q][eval_type+'-aulc'])
            if c != q:
                test = wilcoxon(dic[d][c][eval_type+'-aulc'], dic[d][q][eval_type+'-aulc'], alternative='less')
                if test.pvalue < 0.001:
                    test_results[q_idx, d_idx] = 3
                else:
                    test = wilcoxon(dic[d][c][eval_type+'-aulc'], dic[d][q][eval_type+'-aulc'], alternative='greater')
                    if test.pvalue < 0.001:
                        test_results[q_idx, d_idx] = -3

        aulc_vals_d = np.array(aulc_vals_d)
        aulc_vals.append(aulc_vals_d)
        aulc_ranks[:, d_idx] = np.mean(
            np.array([rankdata(aulc_vals_d[:, r]) for r in range(np.size(aulc_vals_d, axis=1))]),
            axis=0)
    return np.array(aulc_vals), aulc_ranks, test_results
