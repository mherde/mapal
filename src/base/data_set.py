import numpy as np

from sklearn.utils import check_array, check_consistent_length


class DataSet(object):
    """DataSet

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    n_annotators: int
        Number of annotators.
    y: array-like, shape (n_samples, n_annotators_)
        Class labels of the given samples X.
    c: array-like, shape (n_samples)
        Confidence scores for labeling the given samples X.

    Attributes
    ----------
    X_: numpy.ndarray, shape (n_samples, n_features)
        Samples of the whole data set.
    n_annotators_: int
        Number of annotators.
    y_: numpy.ndarray, shape (n_samples, n_annotators_)
        Class labels of the given samples X.
    c_: numpy.ndarray, shape (n_samples, n_annotators_)
        Confidence scores for labeling the given samples X.
    """

    def __init__(self, X, n_annotators=1, y=None, c=None):
        # set parameters
        self.X_ = check_array(X, copy=True)
        self.y_ = np.full((len(self.X_), n_annotators), np.nan) if y is None else check_array(y, force_all_finite=False,
                                                                                              copy=True)
        self.c_ = np.full((len(self.X_), n_annotators), np.nan) if c is None else check_array(c, force_all_finite=False,
                                                                                              copy=True)
        self.n_annotators_ = np.size(self.y_, axis=1)

        # check parameters
        if n_annotators != self.n_annotators_:
            raise ValueError('n_annotators_ is incompatible to y')
        check_consistent_length(self.X_, self.y_)
        check_consistent_length(self.y_, self.c_)

    def __len__(self):
        """
        Number of all samples in this object.

        Returns
        -------
        n_samples: int
        """
        return len(self.X_)

    def len_labeled(self, per_annotator=False):
        """
        Number of labeled samples in this object.

        Parameters
        ----------
        per_annotator: bool
            If true, the number of labeled samples is returned for each annotator.

        Returns
        -------
        n_labeled_samples : int | array-like
            Total number of labeled samples or number of labeled samples for each annotator.
        """
        if not per_annotator:
            return len(self.get_labeled_indices())
        else:
            return np.asarray([np.sum(~np.isnan(self.y_[:, a_idx])) for a_idx in range(self.n_annotators_)])

    def len_unlabeled(self, per_annotator=False):
        """
        Number of unlabeled samples in this object.

        Parameters
        ----------
        per_annotator: bool
            If true, the number of unlabeled samples is returned for each annotator.

        Returns
        -------
        n_unlabeled_samples : int | array-like
            Total number of unlabeled samples or number of unlabeled samples for each annotator.
        """
        if not per_annotator:
            return len(self.get_unlabeled_indices())
        else:
            return np.asarray([np.sum(np.isnan(self.y_[:, a_idx])) for a_idx in range(self.n_annotators_)])

    def get_labeled_indices(self, annotator_id=None):
        """
        Returns indices of all labeled samples.

        Parameters
        ----------
        annotator_id: None | int
            If an annotator id is specified, indices of labeled samples are returned for that annotator.

        Returns
        -------
        labeled_indices: array-like, shape (n_labeled_samples)
            Indices of labeled samples.
        """
        if annotator_id is None:
            return np.where(~np.isnan(self.y_).all(axis=1))[0]
        else:
            return np.where(~np.isnan(self.y_[:, annotator_id]))[0]

    def get_fully_labeled_indices(self):
        """
        Returns indices of all fully labeled samples.

        Returns
        -------
        fully_labeled_indices: array-like, shape (n_fully_labeled_samples)
            Indices of fully labeled samples.
        """
        return np.where(~np.isnan(self.y_).any(axis=1))[0]

    def get_unlabeled_indices(self, annotator_id=None):
        """
        Returns indices of all unlabeled samples.

        Parameters
        ----------
        annotator_id: None | int
            If an annotator id is specified, indices of unlabeled samples are returned for that annotator.

        Returns
        -------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)
            Indices of unlabeled samples.
        """
        if annotator_id is None:
            return np.where(np.isnan(self.y_).any(axis=1))[0]
        else:
            return np.where(np.isnan(self.y_[:, annotator_id]))[0]

    def get_fully_unlabeled_indices(self):
        """
        Returns indices of all fully unlabeled samples.

        Returns
        -------
        fully_unlabeled_indices: array-like, shape (n_fully_unlabeled_samples)
            Indices of fully unlabeled samples.
        """
        return np.where(np.isnan(self.y_).all(axis=1))[0]

    def update_entries(self, sample_indices, y, c=None):
        """
        Updates labels and confidence scores for given samples X.

        Parameters
        ----------
        sample_indices: array-like, shape (n_samples)
            Indices of samples whose labels and confidence scores are updated.
        y: array-like, shape (n_samples, n_annotators_)
            Class labels of the unlabeled samples X.
        c: array-like, shape (n_samples)
            Confidence scores for labeling the samples X.
        """
        y = check_array(y, force_all_finite=False)

        not_nan = ~np.isnan(y)
        ids = np.zeros_like(self.y_, dtype=bool)
        ids[sample_indices] = not_nan
        self.y_[ids] = y[not_nan]

        if c is not None:
            c = check_array(c, force_all_finite=False)
            self.c_[ids] = c[not_nan]






