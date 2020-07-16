import numpy as np

from src.base.query_strategy import QueryStrategy


class RS(QueryStrategy):
    """RS

    This class implements the random sampling (RS) algorithm such that the samples and annotators are selected randomly.

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

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

    def compute_scores(self):
        """
        Compute score for each sample-annotator-pair. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_samples, n_annotators)
            Score of each each sample-annotator-pair.
        """
        scores = np.zeros_like(self.data_set_.y_)
        is_labeled = ~np.isnan(self.data_set_.y_)
        scores[is_labeled] = np.nan
        return scores
