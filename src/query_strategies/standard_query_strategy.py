import numpy as np

from src.base.query_strategy import QueryStrategy
from src.query_strategies.random_sampling import RS


class StandardQueryStrategy(QueryStrategy):
    """StandardQueryStrategy

    This class implements a standard (simple) query strategy where samples are selected according to a base strategy,
    e.g., random sampling or uncertainty sampling. Afterwards a user-defined number of annotators are randomly selected
    for labeling the selected sample.

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.
    n_annotators_per_sample: int in {1, ..., n_annotators}
        Fixed number of annotators labeling a selected sample.
    base_strategy: QueryStrategy
        A base strategy selects the sample, e.g., random sampling or uncertainty sampling.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    n_annotators_per_sample_: int in {1, ..., n_annotators}
        Fixed number of annotators labeling a selected sample.
    base_strategy_: QueryStrategy
        A base strategy selects the sample, e.g., random sampling or uncertainty sampling.
    prev_selection_: shape (n_selected_samples, 2)
        Previous selection of samples and annotators. An entry prev_selection_[i, 0] gives the sample index of the i-th
        selected sample, whereas prev_selection_[i, 1] gives the corresponding annotator selected for labeling.
    """

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.n_annotators_per_sample_ = int(kwargs.pop('n_annotators_per_sample', 1))
        if self.n_annotators_per_sample_ < 1 or self.n_annotators_per_sample_ > self.data_set_.n_annotators_:
            raise ValueError(
                "'n_annotators_per_sample' need to be an integer in the interval [1, n_annotators]"
            )

        self.base_strategy_ = kwargs.pop('base_strategy', RS(data_set=self.data_set_,
                                                             random_state=self.random_state_))
        if not isinstance(self.base_strategy_, QueryStrategy):
            raise TypeError(
                "'base_strategy' has to be an object of the class QueryStrategy"
            )

    def compute_scores(self):
        """
        Compute score for each sample-annotator-pair. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_samples, n_annotators)
            Score of each each sample-annotator-pair.
        """
        annotator_cnt = 0
        is_labeled = ~np.isnan(self.data_set_.y_)

        if self.prev_selection_ is not None:
            sample_id = self.prev_selection_[0, 0]
            annotator_cnt = np.sum(~np.isnan(self.data_set_.y_[sample_id])) % self.n_annotators_per_sample_

        if annotator_cnt == 0:
            scores = self.base_strategy_.compute_scores() / (np.sum(is_labeled, axis=0) + 1)
        else:
            scores = np.full(self.data_set_.y_.shape, np.nan)
            scores[self.prev_selection_[0, 0]] = 1 / (np.sum(is_labeled, axis=0) + 1)

        scores[is_labeled] = np.nan
        is_fully_labeled = np.sum(is_labeled, axis=1) == self.n_annotators_per_sample_
        if np.sum(is_fully_labeled) == len(self.data_set_):
            self.n_annotators_per_sample_ += 1
            is_fully_labeled = np.sum(is_labeled, axis=1) == self.n_annotators_per_sample_
        scores[is_fully_labeled] = np.nan
        return scores
