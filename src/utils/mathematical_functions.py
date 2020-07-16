import numpy as np

from itertools import chain, combinations

from sklearn.utils import check_random_state, check_consistent_length, check_array


def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def estimate_bandwidth(n_samples, n_features):
    nominator = 2 * n_samples * n_features
    denominator = (n_samples - 1) * np.log((n_samples - 1) / ((np.sqrt(2) * 10 ** -6) ** 2))
    bandwidth = np.sqrt(nominator / denominator)
    return bandwidth


def rand_arg_max(arr, axis=1, random_state=None):
    """
    Returns index of maximum element per given axis. In case of ties, the index is chosen randomly.

    Parameters
    ----------
    arr: array-like
        Array whose maximum elements' indices are determined.
    axis: int
        Indices of maximum elements are determined along this axis.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Returns
    -------
    max_indices: array-like
        Indices of maximum elements.
    """
    random_state = check_random_state(random_state)
    arr = np.array(arr)
    arr_max = arr.max(axis, keepdims=True)
    tmp = random_state.uniform(low=1, high=2, size=arr.shape) * (arr == arr_max)
    return tmp.argmax(axis)


def rand_arg_min(arr, axis=1, random_state=None):
    """
    Returns index of minimum element per given axis. In case of ties, the index is chosen randomly.

    Parameters
    ----------
    arr: array-like
        Array whose minimum elements' indices are determined.
    axis: int
        Indices of minimum elements are determined along this axis.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Returns
    -------
    min_indices: array-like
        Indices of minimum elements.
    """
    random_state = check_random_state(random_state)
    arr = check_array(arr, ensure_2d=False)
    arr_min = arr.min(axis, keepdims=True)
    tmp = random_state.uniform(low=1, high=2, size=arr.shape) * (arr == arr_min)
    return tmp.argmax(axis)


def compute_vote_vectors(y, c=None, n_unique_votes=None, probabilistic=False):

    # check input class labels
    y = check_array(y, ensure_2d=False, force_all_finite=False, copy=True)
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    is_nan_y = np.isnan(y)
    y[is_nan_y] = 0
    y = y.astype(int)
    n_unique_votes = np.size(np.unique(y), axis=0) if n_unique_votes is None else n_unique_votes

    # check input confidence scores
    c = np.ones_like(y) if c is None else check_array(c, ensure_2d=False, force_all_finite=False, copy=True)
    c = c if c.ndim == 2 else c.reshape((-1, 1))
    check_consistent_length(y, c)
    check_consistent_length(y.T, c.T)
    c[np.logical_and(np.isnan(c), ~is_nan_y)] = 1

    if probabilistic:
        # compute probabilistic votes per class
        n_annotators = np.size(y, axis=1)
        n_samples = np.size(y, axis=0)
        sample_ids = np.arange(n_samples)
        P = np.array([(1 - c) / (n_unique_votes - 1)] * n_unique_votes).reshape(n_annotators, n_samples, n_unique_votes)
        for a in range(n_annotators):
            P[a, sample_ids,  y[:, a]] = c[:, a]
            P[a, is_nan_y[:, a]] = np.nan
        V = np.nansum(P, axis=0)
        #V_sum = V.sum(axis=1, keepdims=True)
        #is_not_zero = (V_sum != 0).flatten()
        #V[is_not_zero] /= V_sum[is_not_zero]
    else:
        # count class labels per class and weight by confidence scores
        c[np.logical_or(np.isnan(c), is_nan_y)] = 0
        y_off = y + np.arange(y.shape[0])[:, None] * n_unique_votes
        V = np.bincount(y_off.ravel(), minlength=y.shape[0] * n_unique_votes, weights=c.ravel())
        V = V.reshape(-1, n_unique_votes)

    return V