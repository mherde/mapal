import numpy as np

from src.base.query_strategy import QueryStrategy
from src.query_strategies.uncertainty_sampling import uncertainty_scores
from src.utils.mathematical_functions import compute_vote_vectors

from sklearn.utils import check_consistent_length, check_array, column_or_1d
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder


class MCID(QueryStrategy):
    """MCID

    Multi-class information density (MCID) is a query strategy which comprises of three components:
    (1) density,
    (2) unknowingness, and
    (3) inconsistency.
    The density component measures how densely samples are positioned around a given  point, and the unknowingness
    component measures how many samples are labeled thus far. The inconsistency component measures how heterogeneous the
    label distributions around a given sample.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples and class labels.
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    n_components: int, optional (default=20)
        Number of components to be fitted by a Gaussian mixture model.
    random_state: numeric | np.random.RandomState | None, optional (default=None)
        Random state for annotator selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples and class labels.
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    y_cluster_: np.ndarray, shape (n_samples)
        Cluster labels of all available samples.
    p_x_: np.ndarray, shape (n_samples)
        Densities of all available samples.
    random_state_: numeric | np.random.RandomState | None, optional (default=None)
        Random state for annotator selection.
    """

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

        n_components = int(kwargs.pop('n_components', np.min([20, len(self.data_set_)])))
        if n_components < 0 or n_components > len(self.data_set_):
            raise ValueError("'n_components' must be an integer in the interval [1, n_samples]")

        # fit Gaussian mixture model for pre-clustering
        gmm = BayesianGaussianMixture(n_components=n_components, covariance_type='spherical', max_iter=1000,
                                      random_state=self.random_state_)
        gmm.fit(self.data_set_.X_)
        self.y_cluster_ = gmm.predict(self.data_set_.X_)
        self.p_x_ = np.exp(gmm.score_samples(self.data_set_.X_))

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
        scores[unlabeled_indices] = information_density_uncertainty(P=P,
                                                                    y_class_labeled=self.data_set_.y_[labeled_indices],
                                                                    p_x=self.p_x_[unlabeled_indices],
                                                                    y_cluster=self.y_cluster_,
                                                                    labeled_indices=labeled_indices,
                                                                    unlabeled_indices=unlabeled_indices).reshape(-1, 1)
        is_labeled = ~np.isnan(self.data_set_.y_)
        scores[is_labeled] = np.nan
        return scores


def information_density_uncertainty(P, p_x, y_cluster, labeled_indices, unlabeled_indices, y_class_labeled):
    """
    Computes information density based uncertainty scores, which require all available samples to be clustered.

    Parameters
    ----------
    P: array-like, shape (n_unlabeled_samples, n_classes)
        Class membership probabilities of each unlabeled sample.
    p_x: array-like, shape (n_unlabeled_samples)
        Density of each unlabeled sample.
    y_cluster: array-like, shape (n_unlabeled_samples)
        Cluster label of each unlabeled sample.
    labeled_indices: array-like, shape (n_labeled_samples)
        Indices of labeled samples
    unlabeled_indices: array-like, shape (n_unlabeled_samples)
        Indices of unlabeled samples.
    y_class_labeled: array-like, shape (n_labeled_samples, n_annotators)
        Class labels of each labeled sample.
    """
    # ----------------------------------------------CHECK PARAMETERS----------------------------------------------------
    P = check_array(P)
    y_cluster = column_or_1d(y_cluster)
    labeled_indices = column_or_1d(labeled_indices)
    unlabeled_indices = column_or_1d(unlabeled_indices)
    y_class_labeled = check_array(y_class_labeled, ensure_min_samples=0, force_all_finite=False)
    check_consistent_length(labeled_indices, y_class_labeled)
    check_consistent_length(P, unlabeled_indices)
    check_consistent_length(unlabeled_indices, p_x)

    # ----------------------------------------------COMPUTE UNCERTAINTIES-----------------------------------------------
    # compute proportion of unlabeled samples in a cluster
    le = LabelEncoder().fit(y_cluster)
    y_cluster = le.transform(y_cluster)
    all_clusters, cluster_cnt = np.unique(y_cluster, return_counts=True)
    cluster_weights = np.zeros(len(all_clusters))
    cluster_weights[all_clusters] = 1 / cluster_cnt
    y_cluster_unlabeled = y_cluster[np.setdiff1d(unlabeled_indices, labeled_indices)]
    clusters, n_unlabeled_samples_per_cluster = np.unique(y_cluster_unlabeled, return_counts=True)
    cluster_cnt = np.zeros_like(cluster_cnt)
    cluster_cnt[clusters] += n_unlabeled_samples_per_cluster
    cluster_weights *= cluster_cnt

    # compute entropy of class labels in a cluster
    n_classes = np.size(P, axis=1)
    cluster_entropies = np.ones_like(all_clusters) * np.log(n_classes)
    if len(y_class_labeled):
        y_votes_labeled = compute_vote_vectors(y=y_class_labeled, n_unique_votes=n_classes)

        for c in all_clusters:
            y_votes_c = y_votes_labeled[y_cluster[labeled_indices] == c]
            if len(y_votes_c):
                y_votes_c[0] += 1
                P_cluster = np.sum(y_votes_c, axis=0) / np.sum(y_votes_c)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cluster_entropies[c] = np.nansum(-P_cluster * np.log(P_cluster))

    # compute entropy based uncertainty scores
    uncertainties = uncertainty_scores(P=P, method='entropy')

    cluster_weights = np.ones_like(cluster_weights) if np.sum(cluster_weights) == 0 else cluster_weights
    y_cluster_unlabeled = y_cluster[unlabeled_indices]
    return uncertainties * p_x * cluster_weights[y_cluster_unlabeled] * cluster_entropies[y_cluster_unlabeled]
