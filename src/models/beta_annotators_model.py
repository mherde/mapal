import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import KERNEL_PARAMS
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.exceptions import NotFittedError

from src.models.parzen_window_classifier import PWC


class BAM(BaseEstimator):
    """BAM

    The Beta Annotators Model (BAM) estimates the annotation performances of multiple
    annotators per sample. Given several samples and corresponding label vectors of these annotators, the number of
    compliant votes and number of incompliant votes in the neighborhood of a sample are counted for each annotator.
    Together, these votes are used as inputs to a Beta distribution.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    prior: array-like, shape (2)
        Prior observations of compliant votes and incompliant votes in the neighborhood of a sample.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int, optional (default=None)
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    weights_type: 'entropy' | 'margin'
        Type of weights to be computed.
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    kwargs: dict,
        Any further parameters are passed directly to the metric/kernel function.

    Attributes
    ----------
    n_classes_: int
        Number of classes.
    prior_: array-like, shape (2)
        Prior observations of compliant votes and incompliant votes in the neighborhood of a sample.
    metric_: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors_: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    weights_type_: 'entropy' | 'margin'
        Type of weights to be computed.
    random_state_: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    kwargs_: dict,
        Any further parameters are passed directly to the kernel function.
    pwc_list_: array-like, shape (n_annotators)
        For each annotator one fitted Parzen Window Classifier [2] used to estimate the annotation performance.
    n_annotators_: int
        Number of annotators determined when fitting.
    """
    def __init__(self, n_classes, weights_type='entropy', metric='rbf', n_neighbors=None, prior=None, random_state=None, **kwargs):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 1:
            raise ValueError("'n_classes' must be an integer >= 2")

        self.prior_ = np.array([1, 0.1]) if prior is None else check_array(prior, ensure_2d=False)
        if self.prior_.shape[0] != 2:
            raise ValueError("'prior' must have shape (2)")

        self.n_neighbors_ = int(n_neighbors) if n_neighbors is not None else n_neighbors
        if self.n_neighbors_ is not None and self.n_neighbors_ < 1:
            raise ValueError("'n_neighbors' must be a positive integer or None")

        self.weights_type_ = str(weights_type)
        if self.weights_type_ not in ['entropy', 'margin']:
            raise ValueError("'weights_type' must be in {}".format(['entropy', 'margin']))

        self.metric_ = metric
        if self.metric_ not in KERNEL_PARAMS.keys() and self.metric_ != PWC.PRECOMPUTED:
            raise ValueError("'metric' must be in {}".format(KERNEL_PARAMS.keys()))

        self.random_state_ = check_random_state(random_state)
        self.kwargs_ = kwargs
        self.pwc_list_ = None
        self.n_annotators_ = None

    def fit(self, X, y, c=None):
        """
        Given the labels of multiple annotators, this method fits the annotators model to estimate annotation
        performances, i.e. label accuracies, of these multiple annotators.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples, n_annotators)
            Labels provided by multiple annotators. An entry y[i, j] indicates that the annotator with index j has not
            provided a label for the sample with index i.
        """
        # check input parameters
        X = check_array(X)
        y = check_array(y, force_all_finite=False)
        check_consistent_length(X, y)
        c = np.ones_like(y) if c is None else check_array(c, force_all_finite=False)

        # determine number of annotators
        self.n_annotators_ = np.size(y, axis=1)
        is_labeled = ~np.isnan(y)

        # fit PWC per annotator
        self.pwc_list_ = []
        pwc = PWC(n_classes=self.n_classes_, metric=self.metric_, n_neighbors=self.n_neighbors_,
                  random_state=self.random_state_, **self.kwargs_)
        for a in range(self.n_annotators_):
            mask = list(range(self.n_annotators_))
            mask.remove(a)
            pwc.fit(X, y[:, mask], c[:, mask])
            pwc_a = PWC(n_classes=2, metric=self.metric_, n_neighbors=self.n_neighbors_,
                        random_state=self.random_state_, **self.kwargs_)
            if np.sum(is_labeled[:, a]):
                X_a = X[is_labeled[:, a]]
                y_a = np.asarray(y[is_labeled[:, a], a], dtype=int)
                P_a = pwc.predict_proba(X_a, normalize=False) + 1
                P_a /= np.sum(P_a, axis=1, keepdims=True)
                y_mv = np.argmax(P_a, axis=1)
                y_votes_a = np.zeros((len(y_a), 1))
                is_false = y_mv != y_a
                y_votes_a[is_false, 0] = 1
                c_votes_a = np.zeros_like(y_votes_a)
                if self.weights_type_ == 'entropy':
                    c_votes_a[:, 0] = 1 - np.sum(-np.log(P_a) * P_a, axis=-1)/np.log(self.n_classes_)
                elif self.weights_type_ == 'margin':
                    P_sort = np.sort(P_a, axis=1)
                    c_votes_a[:, 0] = P_sort[:, -1] - P_sort[:, -2]
                pwc_a.fit(X_a, y=y_votes_a, c=c_votes_a)
            self.pwc_list_.append(pwc_a)

        return self

    def predict_k_vectors(self, X):
        """
        Computes k-vectors k_a_x=[n_compliant_votes_a_x, n_incompliant_votes_a_x] for all
        annotators a=1 , ..., n_annotators and all samples x=1, ..., n_samples.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Samples on which k-vectors are predicted.

        Returns
        -------
        K: array-like, shape (n_annotators, n_samples, 2)
            Predicted k-vectors.
        """
        if self.pwc_list_ is None:
            raise NotFittedError("This BAM instance is not fitted yet. Call 'fit' with appropriate "
                                 "arguments before using this estimator.")
        X = check_array(X)
        K = np.zeros((self.n_annotators_, len(X), 2))
        for a in range(self.n_annotators_):
            K[a] = self.pwc_list_[a].predict_proba(X, normalize=False)
        return K

    def predict_proba(self, X):
        """
        Computes probabilities that an annotator provides the correct/false annotation for each sample in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Samples on which probabilities are predicted.

        Returns
        -------
        P: array-like, shape (n_annotators, n_samples, 2)
            P[a, x, 0] is the probability that annotator provides the correct annotation for sample x.
        """
        P = self.predict_k_vectors(X=X) + self.prior_
        P /= np.sum(P, axis=-1, keepdims=1)
        return P
