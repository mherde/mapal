import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array

from src.utils.mathematical_functions import rand_arg_max, compute_vote_vectors


class ExpEst(BaseEstimator):
    """ExpEst

    Expertise Estimation (ExpEst) [1] estimates the class-sensitive annotation performances, i.e. label accuracies,
    of multiple annotators. Given several label vectors of these annotators, the majority vote per vector is computed.
    Subsequently, the individual labels of the annotators are compared to the majority vote for the class-sensitive
    label accuracy for each annotator-class pair.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    Attributes
    ----------
    n_classes_: int
        Number of classes.
    random_state_: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.

    References
    ----------
    [1] Moon, S., & Carbonell, J. G. (2014). Proactive learning with multiple class-sensitive labelers.
        2014 International Conference on Data Science and Advanced Analytics (DSAA), 32–38.
        https://doi.org/10.1109/DSAA.2014.7058048
    """

    def __init__(self, n_classes, alpha=0.05, random_state=None):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 0:
            raise ValueError("'n_classes' must be a positive integer")
        self.random_state_ = check_random_state(random_state)

    def predict(self, y, c=None):
        """
        Given the labels of multiple annotators, this method estimates the annotation performances, i.e. label
        accuracies, of these multiple annotators per class.

        Parameters
        ----------
        y: array-like, shape (n_samples, n_annotators)
            Labels provided by multiple annotators. An entry y[i, j]=np.nan indicates that the annotator with index j
            has not provided a label for the sample with index i.
        c: array-like, shape (n_samples, n_annotators)
            Weights for the individual labels.
            Default is c[i, j]=1 as weight for the label entry y[i, j].

        Returns
        -------
        annot_acc: numpy.ndarray, shape (n_annotators, n_classes)
            The entry annot_acc[a, c] indicates the mean labeling accuracy of annotator a for instances of class c.
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
        annot_acc = np.ones((n_annotators, self.n_classes_))
        for a in range(n_annotators):
            y_mv_a = y_mv[is_labeled[:, a]]
            is_correct = np.equal(y_mv_a, y[is_labeled[:, a], a])
            for c in range(self.n_classes_):
                is_c = y_mv_a == c
                annot_acc[a, c] = np.mean(is_correct[is_c]) if np.sum(is_c) else 1

        return annot_acc

