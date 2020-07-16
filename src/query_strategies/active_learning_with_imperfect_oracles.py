import numpy as np

from copy import deepcopy

from src.base.query_strategy import QueryStrategy
from src.models.error_probs_model import ErrorProbsModel
from src.query_strategies.uncertainty_sampling import uncertainty_scores


class ALIO(QueryStrategy):
    """ALIO

    Parameters
    ----------
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    label_proportion: float in intervall (0, 1)
        Proportion of samples to be labeled by all available annotators.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.
    kwargs: dict,
        Any further parameters are passed directly to the metric/kernel function.

    Attributes
    ----------
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    min_samples_: int
        Minimal number of samples to be labeled by all available annotators.
    error_probs_model_: ErrorProbsModel
        The annotators model to estimate the sample-dependent error probabilities of the annotators.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Huang, S. J., Chen, J. L., Mu, X., & Zhou, Z. H. (2017). Cost-effective Active Learning from Diverse Labelers.
        Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17), 1879–1885.
        Melbourne, Australia.
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), random_state=kwargs.pop('random_state', None))
        # check number of classes
        self.n_classes_ = int(kwargs.pop('n_classes'))
        if self.n_classes_ <= 1:
            raise ValueError("'n_classes' must be an integer greater than one")

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

        # create annotator model
        self.error_probs_model_ = ErrorProbsModel(n_classes=self.n_classes_, random_state=deepcopy(self.random_state_),
                                                  **kwargs)

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
        self.clf_.fit(X=self.data_set_.X_[labeled_indices], y=self.data_set_.y_[labeled_indices],
                      c=self.data_set_.c_[labeled_indices])

        # storage for scores
        scores = np.full((len(self.data_set_), self.data_set_.n_annotators_), np.nan)

        fully_labeled_indices = self.data_set_.get_fully_labeled_indices()
        if len(fully_labeled_indices) < self.min_samples_:
            # exploration of annotation performances

            # count number of annotator labeled last selected sample
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
            uncertainties = uncertainty_scores(P=P, method='entropy').reshape(-1, 1)
            scores[unlabeled_indices] = 1/uncertainties

            if self.annot_perfs_ is None:
                # retrain annotators model
                self.error_probs_model_.fit(X=self.data_set_.X_[fully_labeled_indices],
                                            y=self.data_set_.y_[fully_labeled_indices],
                                            c=self.data_set_.c_[fully_labeled_indices])

                # estimate annotation performances per unlabeled sample and annotator
                self.annot_perfs_ = self.error_probs_model_.predict(self.data_set_.X_)

            # combine uncertainties with annotation performance estimates
            if self.annot_perfs_ is not None:
                scores[unlabeled_indices] *= self.annot_perfs_[unlabeled_indices]

        # mask out already selected sample-annotator-pairs
        is_labeled = ~np.isnan(self.data_set_.y_)
        scores[is_labeled] = np.nan

        return -scores
