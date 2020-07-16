import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_random_state, check_array, check_consistent_length

from src.utils.mathematical_functions import rand_arg_min, rand_arg_max, compute_vote_vectors


class PWC(BaseEstimator, ClassifierMixin):
    """PWC

    The Parzen Window Classifier (PWC) is a simple and probabilistic classifier [1]. This classifier is based on a
    non-parametric density estimation obtained by applying a kernel function.

    Parameters
    ----------
    n_classes: int,
        This parameter indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    probabilistic: boolean, optional (default=False)
            Flag whether confidences are interpreted as probabilities.
    combine_labels: boolean, optional (default=False)
        Flag whether labels are combined per sample based on (weighted majority vote).
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on class labels in case of ties.
    kwargs: dict,
        Any further parameters are passed directly to the metric/kernel function.

    Attributes
    ----------
    n_classes_: int,
        This parameters indicates the number of available classes.
    metric_: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors_: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    probabilistic_: boolean, optional (default=False)
            Flag whether confidences are interpreted as probabilities.
    random_state_: None | int | numpy.random.RandomState
        The random state used for deciding on class labels in case of ties.
    kwargs_: dict,
        Any further parameters are passed directly to the kernel function.
    X_: array-like, shape (n_samples, n_features)
        The sample matrix X is the feature matrix representing the samples.
    V_: array-like, shape (n_samples, n_classes)
        The class labels are represented by counting vectors. An entry V[i,j] indicates how many class labels of class j
        were provided for training sample x_i.

    References
    ----------
    [1] O. Chapelle, "Active Learning for Parzen Window Classifier",
        Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    PRECOMPUTED = 'precomputed'

    def __init__(self, n_classes, metric='rbf', n_neighbors=None, probabilistic=False,
                 combine_labels=False, random_state=None, **kwargs):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 0:
            raise ValueError("The parameter 'n_classes' must be a positive integer.")

        self.metric_ = str(metric)
        if self.metric_ not in KERNEL_PARAMS.keys() and self.metric_ != PWC.PRECOMPUTED:
            raise ValueError("The parameter 'metric' must be a in {}".format(KERNEL_PARAMS.keys()))

        self.n_neighbors_ = int(n_neighbors) if n_neighbors is not None else n_neighbors
        if self.n_neighbors_ is not None and self.n_neighbors_ <= 0:
            raise ValueError("The parameter 'n_neighbors' must be a positive integer.")

        self.probabilistic_ = probabilistic
        if not isinstance(self.probabilistic_, bool):
            raise TypeError("combine_labels must be boolean")

        self.combine_labels_ = combine_labels
        if not isinstance(self.combine_labels_, bool):
            raise TypeError("combine_labels must be boolean")

        self.random_state_ = check_random_state(random_state)
        self.kwargs_ = kwargs
        self.X_ = None
        self.y_ = None
        self.V_ = None

    def fit(self, X, y, c=None):
        """
        Fit the PWC using X as training data, y as class labels, and c as optional confidence scores.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples) or (n_samples, n_annotators)
            It contains the class labels of the training sample, which were provided by the annotators.
            The number of class labels may be variable for the samples, where missing labels are represented by np.nan.
        c: array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the annotators' confidence scores for labeling the training samples.
            Missing confidence scores are represented by np.nan.

        Returns
        -------
        self: PWC,
            The PWC is fitted on the training data.
        """
        if np.size(X) > 0:
            # check input samples
            self.X_ = check_array(X)
            check_consistent_length(self.X_, y)
            self.V_ = compute_vote_vectors(y=y, c=c, n_unique_votes=self.n_classes_,
                                           probabilistic=self.probabilistic_)

            # combine class labels of training samples (majority vote weighted by confidence scores)
            if self.combine_labels_:
                temp = np.zeros_like(self.V_)
                temp[np.arange(len(self.V_)), rand_arg_max(self.V_, random_state=self.random_state_, axis=1)] = 1
                zero_rows = np.where(~self.V_.any(axis=1))[0]
                temp[zero_rows] = 0
                self.V_ = temp

        return self

    def predict_proba(self, X, normalize=True):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        P: array-like, shape (t_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """

        # no training data -> random prediction
        if self.X_ is None or np.size(self.X_, 0) == 0:
            if normalize:
                return np.full((np.size(X, 0), self.n_classes_), 1. / self.n_classes_)
            else:
                return np.zeros((np.size(X, 0), self.n_classes_))

        # calculating metric matrix
        if self.metric_ == PWC.PRECOMPUTED:
            K = X
        else:
            K = pairwise_kernels(X, self.X_, metric=self.metric_, **self.kwargs_)

        if self.n_neighbors_ is None:
            # calculating labeling frequency estimates
            P = K @ self.V_
        else:
            if np.size(self.X_, 0) < self.n_neighbors_:
                n_neighbors = np.size(self.X_, 0)
            else:
                n_neighbors = self.n_neighbors_
            indices = np.argpartition(K, -n_neighbors, axis=1)[:, -n_neighbors:]
            P = np.empty((np.size(X, 0), self.n_classes_))
            for i in range(np.size(X, 0)):
                P[i, :] = K[i, indices[i]] @ self.V_[indices[i], :]

        if normalize:
            # normalizing probabilities of each sample
            normalizer = np.sum(P, axis=1)
            P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
            P[normalizer == 0, :] = [1 / self.n_classes_] * self.n_classes_

        return P

    def predict(self, X, C=None):
        """
        Return class label predictions for the test data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.
        C: array-like, shape (n_classes, n_classes)
            Classification cost matrix.

        Returns
        -------
        y: array-like, shape = [n_samples]
            Predicted class labels class.
        """
        if C is None:
            C = np.ones((self.n_classes_, self.n_classes_))
            np.fill_diagonal(C, 0)

        P = self.predict_proba(X, normalize=True)
        return rand_arg_min(arr=np.dot(P, C), axis=1, random_state=self.random_state_)