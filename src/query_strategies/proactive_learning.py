import numpy as np

from copy import deepcopy

from scipy.stats import multivariate_normal

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

from src.base.query_strategy import QueryStrategy
from src.models.expertise_estimation import ExpEst
from src.query_strategies.multi_class_information_density import information_density_uncertainty
from src.utils.mathematical_functions import estimate_bandwidth

class Proactive(QueryStrategy):
    """Proactive

    Proactive learning with multiple class-sensitive labelers [1] is a query strategy that estimates the label accuracy
    of each annotator per class. The multi-class information density measure is used to estimate the sample utility.
    After an initial exploration phase, where each annotator labels a selected sample, both measures are multiplied to
    jointly select sample-annotator pairs.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    n_components: int, optional (default=20)
        Number of components to be fitted by a Gaussian mixture model.
    label_proportion: float in intervall (0, 1)
        Proportion of samples to be labeled by all available annotators.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    min_samples_: int
        Minimum number of samples to be labeled by all available annotators.
    exp_est_: ExpEst
        The expertise estimation model to estimate the class-dependent label accuracy of the annotators.
    annot_perfs_: np.ndarray, shape (n_annotators, n_classes)
        Class dependent annotation performance of each annotator.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Moon, S., & Carbonell, J. G. (2014). Proactive learning with multiple class-sensitive labelers.
        2014 International Conference on Data Science and Advanced Analytics (DSAA), 32–38.
        https://doi.org/10.1109/DSAA.2014.7058048
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        # check number of classes
        self.n_classes_ = int(kwargs.pop('n_classes', None))
        if self.n_classes_ <= 0:
            raise ValueError("'n_classes' must be a positive integer")

        # check classifier
        self.clf_ = kwargs.pop('clf', None)
        if self.clf_ is None:
            raise TypeError(
                "missing required keyword-only argument 'clf'"
            )
        if not callable(getattr(self.clf_, 'fit', None)) or not callable(
                (getattr(self.clf_, 'predict_proba', None))):
            raise TypeError("'clf' must be an instance with the methods 'fit' and 'predict_proba'")

        # compute minimal number of fully labeled samples
        label_proportion = float(kwargs.pop('label_proportion', 0.05))
        if label_proportion > 0 and label_proportion < 1:
            self.min_samples_ = np.ceil(label_proportion * len(self.data_set_))
        else:
            self.min_samples_ = np.ceil(label_proportion)

        # check number of components of the Gaussian mixture model
        n_components = int(kwargs.pop('n_components', np.min([20, len(self.data_set_)])))
        if n_components <= 0 or n_components > len(self.data_set_):
            raise ValueError("'n_components' must be an integer in the interval [1, n_samples]")
        sigma = estimate_bandwidth(n_samples=self.data_set_.X_.shape[0], n_features=self.data_set_.X_.shape[1])
        gamma = 0.5 * (sigma ** (-2))

        # fit Gaussian mixture model for pre-clustering
        kmeans = KMeans(n_clusters=n_components, random_state=deepcopy(self.random_state_))

        # Cluster the data.
        kmeans.fit(self.data_set_.X_)
        self.y_cluster_ = kmeans.predict(self.data_set_.X_)

        centers = kmeans.cluster_centers_
        P_k = np.ones(n_components) / float(n_components)

        dis = rbf_kernel(self.data_set_.X_, kmeans.cluster_centers_, gamma=gamma)

        # EM percedure to estimate the prior
        max_iter = 1000
        for _ in range(max_iter):
            # E-step P(k|x)
            temp = dis * np.tile(P_k, (len(self.data_set_), 1))
            P_k_x = temp / np.tile(np.sum(temp, axis=1), (n_components, 1)).T

            # M-step
            P_k = 1. / len(self.data_set_) * np.sum(P_k_x, axis=0)

        p_x_k = np.zeros((len(self.data_set_), n_components))
        for i in range(n_components):
            p_x_k[:, i] = multivariate_normal.pdf(self.data_set_.X_, mean=centers[i],
                                                  cov=np.ones(self.data_set_.X_.shape[1]) * sigma**2)

        self.p_x_ = np.dot(p_x_k, P_k).reshape(-1)

        # create expertise estimator
        self.exp_est_ = ExpEst(n_classes=self.n_classes_, random_state=deepcopy(self.random_state_))

        # initialize annotation performances
        self.annot_perfs_ = None

    def compute_scores(self):
        """
        Computes score for each sample-annotator-pair. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_samples, n_annotators)
            Score of each each sample-annotator-pair.
        """
        # retrain classifier
        labeled_indices = self.data_set_.get_labeled_indices()
        y_class_labeled = self.data_set_.y_[labeled_indices]
        self.clf_.fit(X=self.data_set_.X_[labeled_indices], y=self.data_set_.y_[labeled_indices],
                      c=self.data_set_.c_[labeled_indices])

        # storage for scores
        scores = np.full((len(self.data_set_), self.data_set_.n_annotators_), np.nan)

        fully_labeled_indices = self.data_set_.get_fully_labeled_indices()
        if len(fully_labeled_indices) < self.min_samples_:
            # exploration of annotation performances

            # count number of annotators labeled last selected sample
            annotator_cnt = np.sum(~np.isnan(self.data_set_.y_[self.prev_selection_[
                0, 0]])) if self.prev_selection_ is not None else self.data_set_.n_annotators_

            if annotator_cnt < self.data_set_.n_annotators_:
                # relabel last sample
                scores[self.prev_selection_[0, 0]] = 1
            else:
                # select a new sample
                unlabeled_indices = self.data_set_.get_unlabeled_indices()
                scores[unlabeled_indices] = np.ones((len(unlabeled_indices), self.data_set_.n_annotators_))

        else:
            # exploitation of annotation performances

            # get indices of unlabeled samples
            fully_unlabeled_indices = self.data_set_.get_fully_unlabeled_indices()
            unlabeled_indices = self.data_set_.get_unlabeled_indices()
            unlabeled_indices = fully_unlabeled_indices if len(fully_unlabeled_indices) > 0 else unlabeled_indices

            # compute uncertainties for unlabeled samples
            P = self.clf_.predict_proba(self.data_set_.X_[unlabeled_indices])
            uncertainties = information_density_uncertainty(p_x=self.p_x_[unlabeled_indices],
                                                            y_cluster=self.y_cluster_,
                                                            labeled_indices=labeled_indices,
                                                            unlabeled_indices=unlabeled_indices,
                                                            y_class_labeled=y_class_labeled,
                                                            P=P)

            if self.annot_perfs_ is None:
                # estimate annotation performances per unlabeled sample and annotator
                self.annot_perfs_ = self.exp_est_.predict(self.data_set_.y_[labeled_indices])

            # combine uncertainties with annotation performance estimates
            annot_perfs_per_sample = np.sum(P * self.annot_perfs_[:, None], axis=2).T
            scores[unlabeled_indices] = annot_perfs_per_sample * uncertainties[:, None]

        # mask out already selected sample-annotator-pairs
        is_labeled = ~np.isnan(self.data_set_.y_)
        scores[is_labeled] = np.nan

        return scores
