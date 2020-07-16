import numpy as np

from src.base.query_strategy import QueryStrategy
from src.models.interval_estimation_learning import IELearning
from src.query_strategies.uncertainty_sampling import uncertainty_scores
from src.utils.mathematical_functions import rand_arg_max


class IEThresh(QueryStrategy):
    """IEThresh

    The strategy 'Interval Estimation Threshold' (IEThresh) [1] is useful for addressing the exploration vs.
    exploitation trade-off when dealing with multiple error-prone annotators in active learning.
    This class relies on 'Interval Estimation Learning' (IELearning) for estimating the annotation performances, i.e.
    label accuracies, of multiple annotators. Samples are selected based on 'Uncertainty Sampling' (US).
    The selected samples are labeled by the annotators whose estimated annotation performances are equal or greater than
    an adaptive threshold.

    Parameters
    ----------
    data_set: DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    epsilon: float in [0, 1], optional (default=0.1)
        Parameter for specifying the adaptive threshold used for annotator selection.
    n_classes: int
        Number of classes.
    alpha: float in interval (0, 1)
        Half of the confidence level for student's t-distribution.
        Default is 0.05
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    Attributes
    ----------
    data_set_: DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    epsilon_: float in [0, 1], optional (default=0.1)
        Parameter for specifying the adaptive threshold used for annotator selection.
    ie_learning_: IELearning
        Instance of IELearning to estimate the annotation performances.
    selected_annots_: list, shape (n_selected_annotators)
        List of annotators to be selected in the next active learning queries.
    random_state_: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        # check classifier
        self.clf_ = kwargs.pop('clf', None)
        if self.clf_ is None:
            raise TypeError(
                "missing required keyword-only argument 'clf'"
            )
        if not callable(getattr(self.clf_, 'fit', None)) or not callable(
                (getattr(self.clf_, 'predict_proba', None))):
            raise TypeError("'clf' must be an instance with the methods 'fit' and 'predict_proba'")

        # check threshold parameter epsilon
        self.epsilon_ = float(kwargs.pop('epsilon', 0.9))
        if self.epsilon_ < 0 or self.epsilon_ > 1:
            raise ValueError("'epsilon' must be in the interval [0, 1]")

        # create IELearning instance from given parameters
        alpha = kwargs.pop('alpha', 0.05)
        n_classes = kwargs.pop('n_classes', None)
        self.ie_learning_ = IELearning(n_classes=n_classes, alpha=alpha, random_state=self.random_state_)

        self.selected_annots_ = []

    def compute_scores(self):
        # storage for scores
        scores = np.full((len(self.data_set_), self.data_set_.n_annotators_), np.nan)

        if len(self.selected_annots_) == 0 or self.prev_selection_ is None:
            # select a new unlabeled sample

            # reset annotator list
            self.selected_annots_ = []

            # retrain classifier
            labeled_indices = self.data_set_.get_labeled_indices()
            self.clf_.fit(X=self.data_set_.X_[labeled_indices], y=self.data_set_.y_[labeled_indices],
                          c=self.data_set_.c_[labeled_indices])

            # get indices of unlabeled samples
            fully_unlabeled_indices = self.data_set_.get_fully_unlabeled_indices()
            unlabeled_indices = self.data_set_.get_unlabeled_indices()
            unlabeled_indices = fully_unlabeled_indices if len(fully_unlabeled_indices) > 0 else unlabeled_indices

            # compute uncertainty scores to select sample
            P = self.clf_.predict_proba(self.data_set_.X_[unlabeled_indices])
            uncertainties = uncertainty_scores(P=P, method='lc')
            sample_id = unlabeled_indices[rand_arg_max(uncertainties, random_state=self.random_state_, axis=0)]
            is_labeled = ~np.isnan(self.data_set_.y_[sample_id]).flatten()

            # compute annotation performances to select annotators
            annot_perfs = self.ie_learning_.predict(self.data_set_.y_, self.data_set_.c_)[:, -1]
            print(annot_perfs)
            annot_perfs[is_labeled] = np.nan
            with np.errstate(invalid='ignore'):
                n_selected = np.sum(annot_perfs >= self.epsilon_ * np.max(annot_perfs))
            sorted_ids = np.argsort(-annot_perfs)
            selected_ids = sorted_ids[1:n_selected].tolist()
            self.selected_annots_.extend(selected_ids)

            # set score for selected sample and estimated annotation performances
            scores[sample_id, sorted_ids[0]] = 1

        else:
            # select previous sample to be labeled by another annotator
            sample_id = self.prev_selection_[0, 0]
            annot_id = self.selected_annots_.pop(0)
            scores[sample_id, annot_id] = 1

        return scores





