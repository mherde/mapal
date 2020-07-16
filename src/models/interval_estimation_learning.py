import numpy as np

from scipy.stats import t

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array

from src.utils.mathematical_functions import rand_arg_max, compute_vote_vectors


class IELearning(BaseEstimator):
    """IELearning

    Interval Estimation (IE) Learning [1] is useful for addressing the exploration vs. exploitation trade-off.
    This class implements it for estimating the annotation performances, i.e. label accuracies, of multiple annotators.
    Given several label vectors of these annotators, the majority vote per vector is computed.
    Subsequently, the individual labels of the annotators are compared to the majority vote for computing upper
    confidence interval of the annotators' label accuracies.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    alpha: float in interval (0, 1)
        Half of the confidence level for student's t-distribution.
        Default is 0.05
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    Attributes
    ----------
    n_classes_: int
        Number of classes.
    alpha_: float in interval (0, 1)
        Half of the confidence level for student's t-distribution.
        Default is 0.05
    random_state_: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    References
    ----------
    [1] Donmez, P., Carbonell, J. G., & Schneider, J. (2009).
        Efficiently Learning the Accuracy of Labeling Sources for Selective Sampling.
        Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 259–268.
        https://doi.org/10.1145/1557019.1557053
    """

    def __init__(self, n_classes, alpha=0.05, random_state=None):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 0:
            raise ValueError("'n_classes' must be a positive integer")
        self.alpha_ = float(alpha)
        if self.alpha_ <= 0 or self.alpha_ >= 1:
            raise ValueError("'alpha' must be in the interval (0, 1)")
        self.random_state_ = check_random_state(random_state)

    def predict(self, y, c=None):
        """
        Given the labels of multiple annotators, this method estimates the annotation performances, i.e. label
        accuracies, of these multiple annotators.

        Parameters
        ----------
        y: array-like, shape (n_samples, n_annotators)
            Labels provided by multiple annotators. An entry y[i, j] indicates that the annotator with index j has not
            provided a label for the sample with index i.
        c: array-like, shape (n_samples, n_annotators)
            Weights for the individual labels.
            Default is c[i, j]=1 as weight for the label entry y[i, j].

        Returns
        -------
        annot_acc: numpy.ndarray, shape (n_annotators, 3)
            The entry annot_acc[a, 1] indicates the mean labeling accuracy of annotator a while annot_acc[a, 0] and
            annot_acc[a, 2] are the lower and upper bounds for this estimate.
        """
        # determine number of annotators
        y = check_array(y, force_all_finite=False)
        n_annotators = np.size(y, axis=1)

        # flag for labeled entries
        is_labeled = ~np.isnan(y)

        # compute (confidence weighted majority) vote
        V = compute_vote_vectors(y=y, c=c, n_unique_votes=self.n_classes_)
        y_mv = rand_arg_max(arr=V, axis=1, random_state=self.random_state_)

        # compute annotation accuracy estimates
        annot_acc = np.zeros((n_annotators, 3))
        for a_idx in range(n_annotators):
            is_correct = np.equal(y_mv[is_labeled[:, a_idx]], y[is_labeled[:, a_idx], a_idx])
            is_correct = np.concatenate((is_correct, [0, 1]))
            mean = np.mean(is_correct)
            std = np.std(is_correct)
            t_value = t.isf([self.alpha_ / 2], len(is_correct) - 1)[0]
            t_value *= std / np.sqrt(len(is_correct))
            annot_acc[a_idx, 0] = mean - t_value
            annot_acc[a_idx, 1] = mean
            annot_acc[a_idx, 2] = mean + t_value

        return annot_acc

