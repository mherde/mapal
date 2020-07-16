import numpy as np

from src.base.query_strategy import QueryStrategy
from src.models.interval_estimation_learning import IELearning
from src.query_strategies.uncertainty_sampling import uncertainty_scores
from src.utils.mathematical_functions import rand_arg_max, powerset

from scipy.stats import rankdata


class IEAdjCost(QueryStrategy):
    """IEAdjCost

    Interval Estimates with Adjusted Cost (IEAdjCost) is an active learning strategy that learns to select
    the best annotators and samples for labeling. For sample selection, it relies on uncertainty sampling,
    while for annotator selection the interval estimation learning is used.

    Parameters
    ----------
    data_set: DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    n_classes: int
        Number of classes.
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    epsilon: float in [0, 1], optional (default=0.9)
        Parameter for specifying the adaptive threshold used for determining the set O_r (annot_r).
    lmbda: float in (0, 1], optional (default=0.4)
        Fraction of annotators to be added to the set O_l (annot_l).
    delta: float in (0, 1], optional (default=0.3)
        Parameter to control the degree of exploration and used to determine the set O_g (annot_g).
    min_acc: float [0, 1], optional (default=0.95)
        The minimum required accuracy for the annotators in set O_f (annot_f).
    alpha: float in interval (0, 1), optional (default=0.05)
        Half of the confidence level for student's t-distribution.
        Default is 0.05
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    Attributes
    ----------
    data_set_: DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    n_classes_: int
        Number of classes.
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    epsilon_: float in [0, 1], optional (default=0.9)
        Parameter for specifying the adaptive threshold used for determining the set O_r (annot_r).
    min_annots_: int in [0, n_annotators]
        Minimal number of annotators to be added to the set O_l (annot_l).
    delta_: float in (0, 1], optional (default=0.3)
        Parameter to control the degree of exploration and used to determine the set O_g (annot_g).
    min_acc_: float [0, 1], optional (default=0.95)
        The minimum required accuracy for the annotators in set O_f (annot_f).
    selected_annots_: list, shape (n_selected_annotators)
        List of annotators to be selected in the next active learning queries.
    annot_f_: array-like, shape (n_best_annotators)
        Contains the best annotators O_f determined during the exploration phase.
    random_state_: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    References
    ----------
    Zheng, Y., Scott, S., & Deng, K. (2010). Active Learning from Multiple Noisy Labelers with Varied Costs.
    2010 IEEE International Conference on Data Mining, 639–648. https://doi.org/10.1109/ICDM.2010.147
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

        # check annotators fraction parameter lmbda
        lmbda = float(kwargs.pop('lmbda', 0.4))
        if lmbda <= 0 or lmbda > 1:
            raise ValueError("'lmbda' must be in the interval (0, 1]")
        self.min_annots_ = np.ceil(lmbda * self.data_set_.n_annotators_)

        # check delta parameter
        self.delta_ = float(kwargs.pop('delta', 0.3))
        if self.delta_ < 0 or self.delta_ > 1:
            raise ValueError("'delta' must be in the interval (0, 1)")

        # check threshold parameter epsilon
        self.epsilon_ = float(kwargs.pop('epsilon', 0.9))
        if self.epsilon_ < 0 or self.epsilon_ > 1:
            raise ValueError("'epsilon' must be in the interval [0, 1]")

        # check minimum accuracy parameter min_acc
        self.min_acc_ = float(kwargs.pop('min_acc', 0.95))
        if self.min_acc_ < 0 or self.min_acc_ > 1:
            raise ValueError("'min_acc' must be in the interval [0, 1]")

        # create IELearning instance from given parameters
        alpha = kwargs.pop('alpha', 0.05)
        n_classes = kwargs.pop('n_classes', None)
        self.ie_learning_ = IELearning(n_classes=n_classes, alpha=alpha, random_state=self.random_state_)

        self.selected_annots_ = []
        self.annot_f_ = np.zeros(self.data_set_.n_annotators_, dtype=bool)

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

            if np.sum(self.annot_f_) == 0:
                # compute annotation performances and rank them
                annot_perfs = self.ie_learning_.predict(self.data_set_.y_, self.data_set_.c_)
                ranks = rankdata(-annot_perfs[:, 2], method='min')
                print('ranks: {}'.format(ranks))

                # compute annotators whose accuracy estimates sufficiently good
                annot_g = (annot_perfs[:, 2] - annot_perfs[:, 0]) <= self.delta_
                print('O_g: {}'.format(annot_g))

                # compute annotators whose accuracy estimates will be refined
                annot_l = ranks <= self.min_annots_
                annot_l = np.logical_and(annot_l, ~annot_g)
                print('O_l: {}'.format(annot_l))

                # compute annotators used to estimate the majority vote
                annot_r = annot_perfs[:, 2] >= self.epsilon_ * np.max(annot_perfs[:, 2])
                self.data_set_.c_[sample_id, annot_r] = 1
                self.data_set_.c_[sample_id, ~annot_r] = 0
                print('O_r: {}'.format(annot_r))

                # determine set of annotators to be selected
                selected_annots = np.logical_or(annot_l, annot_r)

                # determine set of best annotators
                if np.sum(annot_g) >= self.min_annots_:
                    print('accuracies: {}'.format(annot_perfs[:, 1]))
                    ranks_idx = np.argsort(-rankdata(annot_perfs[:, 1], method='ordinal'))
                    annot_g = annot_g[ranks_idx]
                    ranks_idx = ranks_idx[annot_g]
                    for r in range(len(ranks_idx)):
                        power_sets = powerset(ranks_idx[:r+1])
                        power_sets = [p for p in power_sets if len(p) >= 0.5 * (r+1)]
                        est_acc = 0
                        for p in power_sets:
                            not_p = list(set(ranks_idx) - set(p))
                            est_acc += np.prod(annot_perfs[p, 1]) * np.prod(1-annot_perfs[not_p, 1])
                        if est_acc >= self.min_acc_:
                            self.annot_f_[ranks_idx[:r+1]] = 1
                            selected_annots = self.annot_f_

            else:
                selected_annots = self.annot_f_

                self.data_set_.c_[sample_id, selected_annots] = 1
                self.data_set_.c_[sample_id, ~selected_annots] = 0

            # check whether sample has be already labeled by selected annotators
            selected_annots[is_labeled] = 0
            if np.sum(selected_annots) == 0:
                annot_perfs = self.ie_learning_.predict(self.data_set_.y_, self.data_set_.c_)[:, 1]
                annot_perfs[is_labeled] = -1
                selected_annots[np.argmax(annot_perfs)] = 1
                self.data_set_.c_[sample_id, selected_annots] = 1

            self.selected_annots_.extend(np.argwhere(selected_annots == 1)[:, 0].tolist())

        else:
            # select previous sample to be labeled by another annotator
            sample_id = self.prev_selection_[0, 0]

        annot_id = self.selected_annots_.pop(0)
        scores[sample_id, annot_id] = 1

        return scores
