import numpy as np

from src.base.query_strategy import QueryStrategy

from sklearn.utils import check_array


class US(QueryStrategy):
    """US

    This class implements different variants of the uncertainty sampling (US) algorithm [1]:
     - least confident (lc) [1] and its cost-sensitive version maximal expected misclassification costs [2],
     - smallest margin (sm) [1] and its cost-sensitive version cost-weighted minimum margin [2],
     - and entropy based uncertainty [1].
    In case of multiple annotators, the selected sample is labeled by a randomly chosen annotator.

    Parameters
    ----------
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        Least confidence (lc) queries the sample, whose maximal posterior probability is minimal.
        In case of a given cost matrix, the maximial expected cost variant is used.
        Smallest margin (sm) queries the sample, whose posterior probability gap between
        the most and the second most probable class label is minimal.
        In case of a given cost matrix, the cost-weighted minimum margin is used.
        Entropy ('entropy') queries the sample whose posterior's have the maximal entropy.
        There is no cost-sensitive variant of entropy based uncertainty sampling.
    data_set: base.DataSet
        Data set containing samples and class labels.
    C: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i, j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for least confident ('lc') and smallest margin('sm') variant.
    random_state: numeric | np.random.RandomState | None, optional (default=None)
        Random state for annotator selection.

    Attributes
    ----------
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    method_: {'lc', 'sm', 'entropy'}, optional (default='lc')
        Least confidence (lc) queries the sample whose maximal posterior probability is minimal.
        In case of a given cost matrix, the maximial expected cost variant is used.
        Smallest margin (sm) queries the sample whose posterior probability gap between
        the most and the second most probable class label is minimal.
        In case of a given cost matrix, the cost-weighted minimum margin is used.
        Entropy queries the sample whose posterior's have the maximal entropy.
        There is no cost-sensitive variant of entropy based uncertainty sampling.
    data_set_: base.DataSet
        Data set containing samples and class labels.
    C_: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i, j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for least confident ('lc') and smallest margin('sm') variant.
    prev_selection_: shape (n_selected_samples, 2)
        Previous selection of samples and annotators. An entry prev_selection_[i, 0] gives the sample index of the i-th
        selected sample, whereas prev_selection_[i, 1] gives the corresponding annotator selected for labeling.
    random_state_: numeric | np.random.RandomState | None, optional (default=None)
        Random state for annotator selection.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Chen, P. L., & Lin, H. T. (2013, December). Active learning for multiclass cost-sensitive classification
        using probabilistic models. In 2013 Conference on Technologies and Applications of Artificial Intelligence
        (pp. 13-18). IEEE.
    """

    LC = 'lc'
    SM = 'sm'
    ENTROPY = 'entropy'

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.clf_ = kwargs.get('clf', None)
        if self.clf_ is None:
            raise ValueError(
                "missing required keyword-only argument 'clf'"
            )
        if not callable(getattr(self.clf_, 'fit', None)) or not callable(
                (getattr(self.clf_, 'predict_proba', None))):
            raise TypeError("'clf' must be an instance with the methods 'fit' and 'predict_proba'")

        self.method_ = kwargs.get('method', US.LC)
        if self.method_ not in [US.LC, US.SM, US.ENTROPY]:
            raise ValueError(
                "supported methods are [{}, {}, {}], the given one " "is: {}".format(
                    US.LC, US.SM, US.ENTROPY, self.method_)
            )

        self.C_ = kwargs.pop('C', None)
        if self.C_ is not None:
            self.C_ = check_array(self.C_)
            if np.size(self.C_, axis=0) != np.size(self.C_, axis=1):
                raise ValueError(
                    "C must be a square matrix with shape (n_classes, n_classes)"
                )

    def compute_scores(self):
        """
        Compute score for each sample-annotator-pair. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_samples, n_annotators)
            Score of each each sample-annotator-pair.
        """
        unlabeled_indices = self.data_set_.get_unlabeled_indices()
        labeled_indices = self.data_set_.get_labeled_indices()
        self.clf_.fit(self.data_set_.X_[labeled_indices], self.data_set_.y_[labeled_indices],
                      c=self.data_set_.c_[labeled_indices])
        P = self.clf_.predict_proba(self.data_set_.X_[unlabeled_indices])
        scores = np.zeros_like(self.data_set_.y_)
        scores[unlabeled_indices] = uncertainty_scores(P=P, C=self.C_, method=self.method_).reshape(-1, 1)
        is_labeled = ~np.isnan(self.data_set_.y_)
        scores[is_labeled] = np.nan
        return scores


def uncertainty_scores(P, C=None, method='lc'):
    """
    Computes uncertainty scores. Three methods are available: least confident (lc), smallest margin (sm), and entropy
    based uncertainty. For the least confident and smallest margin methods cost-sensetive variants are implemented in
    case of a given cost matrix.

    Parameters
    ----------
    P: array-like, shape (n_samples, n_classes)
        Class membership probabilities for each sample.
    C: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for least confident variant.
    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        Least confidence (lc) queries the sample whose maximal posterior probability is minimal.
        In case of a given cost matrix, the maximial expected cost variant is used.
        Smallest margin (sm) queries the sample whose posterior probability gap between
        the most and the second most probable class label is minimal.
        In case of a given cost matrix, the cost-weighted minimum margin is used.
        Entropy ('entropy') queries the sample whose posterior's have the maximal entropy.
        There is no cost-sensitive variant of entropy based uncertainty sampling.
    """
    # ----------------------------------------------CHECK PARAMETERS----------------------------------------------------
    # check probabilities P
    P = check_array(P)
    n_classes = P.shape[1]

    # check cost matrix C
    if C is not None:
        C = check_array(C)
        if np.size(C, axis=0) != n_classes or np.size(C, axis=1) != n_classes:
            raise ValueError(
                "'C' must have the shape (n_classes, n_classes)"
            )

    # check method
    if method not in ['lc', 'sm', 'entropy']:
        raise ValueError(
            "supported methods are ['lc', 'sm', 'entropy'], the given one is: {}".format(method)
        )

    # ----------------------------------------------COMPUTE UNCERTAINTIES-----------------------------------------------
    if method == 'lc':
        if C is None:
            return 1 - np.max(P, axis=1)
        else:
            costs = P @ C
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return costs[:, 0]
    elif method == 'sm':
        if C is None:
            P = -(np.partition(-P, 1, axis=1)[:, :2])
            return 1 - np.abs(P[:, 0] - P[:, 1])
        else:
            costs = P @ C
            costs = np.partition(costs, 1, axis=1)[:, :2]
            return -np.abs(costs[:, 0] - costs[:, 1])
    elif method == 'entropy':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nansum(-P * np.log(P), axis=1)