import numpy as np

from abc import ABC, abstractmethod
from src.base.data_set import DataSet

from sklearn.utils import check_random_state


class QueryStrategy(ABC):
    """QueryStrategy

    A query strategy advices on which unlabeled data to be queried next given a pool of labeled and unlabeled data.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    prev_selection_: shape (n_selected_samples, 2)
        Previous selection of samples and annotators. An entry prev_selection_[i, 0] gives the sample index of the i-th
        selected sample, whereas prev_selection_[i, 1] gives the corresponding annotator selected for labeling.
    """

    def __init__(self, data_set, **kwargs):
        self.data_set_ = data_set
        if not isinstance(self.data_set_, DataSet):
            raise TypeError(
                "'data_set' must be an instance of the class 'base.DataSet'"
            )
        self.random_state_ = check_random_state(kwargs.pop('random_state', None))
        self.prev_selection_ = None

    @abstractmethod
    def compute_scores(self):
        """
        Compute score for each sample-annotator-pair. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_samples, n_annotators)
            Score of each each sample-annotator-pair.
        """
        pass

    def make_query(self):
        """
        Returns the indices of the selected samples and the indices of the annotators who shall label the
        selected samples.

        Returns
        -------
        sample_ids: array-like, shape (n_selected_samples)
            The indices of the samples to be labeled.
        annotator_ids: array-like, shape (n_selected_annotators)
            The indices of the annotators who shall label the selected samples.
        """
        # compute score for each sample-annotator-pair
        scores = self.compute_scores()

        # determine sample and annotator indices with maximal scores
        sample_ids, annotator_ids = np.where(scores == np.nanmax(scores))
        random_id = self.random_state_.randint(low=0, high=len(sample_ids))
        sample_ids = np.array([sample_ids[random_id]])
        annotator_ids = np.array(annotator_ids[random_id])

        self.prev_selection_ = np.column_stack((sample_ids, annotator_ids))

        return self.prev_selection_
