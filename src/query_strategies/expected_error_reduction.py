import copy
import numpy as np

from src.base.query_strategy import QueryStrategy

from sklearn.utils import check_array, check_consistent_length


class EER(QueryStrategy):
    """EER

    This class implements the expected error reduction algorithm with different loss functions:
     - log loss (log-loss) [1]
     - and expected misclassification risk (emr) [2].
    In case of multiple annotators, the selected sample is labeled by a randomly chosen annotator.

    Parameters
    ----------
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    method_: {'log-loss', 'emr'}, optional (default='log-loss')
        Variant of expected error reduction to be used: 'log-loss' is cost-insensitive, while 'emr' is a
        cost-sensitive variant.
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    C: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i, j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for 'emr' and 'csl' variant.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    clf_: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    method_: {'log-loss', 'emr'}
        Variant of expected error reduction to be used: 'log-loss' is cost-insensitive, while 'emr' is a
        cost-sensitive variant.
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    C_: array-like, shape (n_classes, n_classes)
        Cost matrix with C[i,j] defining the cost of predicting class j for a sample which actually belongs
        to class i.
    prev_selection_: shape (n_selected_samples, 2)
        Previous selection of samples and annotators. An entry prev_selection_[i, 0] gives the sample index of the i-th
        selected sample, whereas prev_selection_[i, 1] gives the corresponding annotator selected for labeling.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.

    References
    ----------
    [1] Settles, Burr. "Active learning literature survey." University of
        Wisconsin, Madison 52.55-66 (2010): 11.
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. (2010). Breaking the interactive bottleneck in multi-class
        classification with active selection and binary feedback.
        2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2995–3002.
        https://doi.org/10.1109/CVPR.2010.5540047
    """

    EMR = 'emr'
    LOG_LOSS = 'log-loss'

    def __init__(self, **kwargs):
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.n_classes_ = int(kwargs.pop('n_classes', None))
        if self.n_classes_ < 2:
            raise ValueError(
                "'n_classes' must be an integer larger than 1"
            )

        self.clf_ = kwargs.get('clf', None)
        if self.clf_ is None:
            raise TypeError(
                "missing required keyword-only argument 'clf'"
            )
        if not callable(getattr(self.clf_, 'fit', None)) or not callable(
                (getattr(self.clf_, 'predict_proba', None))):
            raise TypeError("'clf' must be an instance with the methods 'fit' and 'predict_proba'")

        self.C_ = check_array(kwargs.get('C', 1 - np.eye(self.n_classes_)))
        if np.size(self.C_, axis=0) != self.n_classes_ or np.size(self.C_, axis=1) != self.n_classes_:
            raise ValueError(
                "'C' must have the shape (n_classes, n_classes)"
            )

        self.method_ = kwargs.get('method', EER.LOG_LOSS)
        if self.method_ not in [EER.EMR, EER.LOG_LOSS]:
            raise ValueError(
                "supported methods are ['{}', '{}'], the given one is: {}".format(EER.EMR, EER.LOG_LOSS, self.method_)
            )

    def compute_scores(self):
        """
        Computes score for each sample-annotator-pair. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_samples, n_annotators)
            Score of each each sample-annotator-pair.
        """
        unlabeled_indices = self.data_set_.get_unlabeled_indices()
        labeled_indices = self.data_set_.get_labeled_indices()
        scores = np.zeros_like(self.data_set_.y_)
        X_labeled = self.data_set_.X_[labeled_indices]
        y_labeled = self.data_set_.y_[labeled_indices]
        X_unlabeled = self.data_set_.X_[unlabeled_indices]
        scores[unlabeled_indices] = expected_error_reduction(clf=self.clf_, X_labeled=X_labeled, y_labeled=y_labeled,
                                                             X_unlabeled=X_unlabeled, C=self.C_,
                                                             method=self.method_).reshape(-1, 1)
        is_labeled = ~np.isnan(self.data_set_.y_)
        scores[is_labeled] = np.nan

        return scores


def expected_error_reduction(clf, X_labeled, y_labeled, X_unlabeled, C=None, method='log-loss'):
    """
    Computes expected error reduction given a classification model in combination with an unlabeled pool and a labeled
    pool of samples. Different methods (including a cost-sensitive one) are available.

    Parameters
    ----------
    clf: classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    X_labeled: array-like, shape (n_labeled_samples, n_features)
        Labeled samples.
    y_labeled: array-like, shape (n_labeled_samples, n_annotators)
        Class labels of labeled samples.
    X_unlabeled: array-like, shape (n_unlabeled_samples, n_features)
        Unlabeled samples.
    C: array-like, shape (n_classes, n_classes), optional (default=1-np.eye(n_classes))
        Cost matrix with C[i,j] defining the cost of predicting class j for a sample with the actual class i.
        Only supported for 'emr' variant.
    method: {'log-loss', 'emr'}, optional (default='log-loss')
        Variant of expected error reduction to be used: 'log-loss' is cost-insensitive, while 'emr' is a
        cost-sensitive variant.

    Returns
    -------
    errors: array-like, shape (n_unlabeled_samples)
        errors[i]=-e_i describes the estimated error e_i after using the unlabeled sample x_i for training.
    """
    # -------------------------------------------CHECK PARAMETERS-------------------------------------------------------
    # check data X_labeled, X_unlabeled, and y_labeled
    X_labeled = check_array(X_labeled, ensure_min_samples=0)
    X_unlabeled = check_array(X_unlabeled)
    y_labeled = check_array(y_labeled, ensure_min_samples=0, force_all_finite=False)
    n_annotators = y_labeled.shape[1]
    check_consistent_length(X_labeled, y_labeled)

    # check method
    if method not in ['emr', 'log-loss']:
        raise ValueError(
            "supported methods are ['emr', 'log-loss'], the given one is: {}".format(method)
        )

    # check clf
    if not callable(getattr(clf, 'fit', None)) or not callable((getattr(clf, 'predict_proba', None))):
        raise TypeError("'clf' must be an instance with the methods 'fit' and 'predict_proba'")
    clf = copy.deepcopy(clf)
    clf.fit(X_labeled, y_labeled)
    P = clf.predict_proba(X_unlabeled)
    n_classes = P.shape[1]

    # check cost matrix C
    C = 1 - np.eye(n_classes) if C is None else check_array(C)
    if np.size(C, axis=0) != n_classes or np.size(C, axis=1) != n_classes:
        raise ValueError(
            "'C' must have the shape (n_classes, n_classes)"
        )

    # -------------------------------------------COMPUTE ERRORS---------------------------------------------------------
    errors = np.zeros(len(X_unlabeled))
    errors_per_class = np.zeros(n_classes)
    for i, x in enumerate(X_unlabeled):
        for yi in range(n_classes):
            y_x = np.full((1, n_annotators), np.nan)
            y_x[0, 0] = yi
            clf.fit(np.vstack((X_labeled, [x])), np.vstack((y_labeled, y_x)))
            if method == 'emr':
                P_new = clf.predict_proba(X_unlabeled)
                costs = np.sum((P_new.T[:, None] * P_new.T).T * C)
            elif method == 'log-loss':
                P_new = clf.predict_proba(X_unlabeled)
                with np.errstate(divide='ignore', invalid='ignore'):
                    costs = -np.nansum(P_new * np.log(P_new))
            errors_per_class[yi] = P[i, yi] * costs
        errors[i] = errors_per_class.sum()
    return -errors
