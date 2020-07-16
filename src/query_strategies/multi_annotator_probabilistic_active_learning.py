import numpy as np

from src.base.query_strategy import QueryStrategy
from src.models.beta_annotators_model import BAM
from src.utils.mathematical_functions import compute_vote_vectors

from sklearn.utils import check_array


class MAPAL(QueryStrategy):

    def __init__(self, **kwargs):
        """MAPAL

        Parameters
        ----------
        model: n_classes
            Number of classes.
        S: array-like, shape (n_samples, n_samples)
            Similarity matrix defining the similarities between all paris of available samples, e.g., S[i,j] describes
            the similarity between the samples x_i and x_j.
            Default similarity matrix is the unit matrix.
        data_set: base.DataSet
            Data set containing samples and class labels.
        random_state: numeric | np.random.RandomState
            Random state for annotator selection.

        Attributes
        ----------
        n_classes_: int
            Number of classes.
        S_: array-like, shape (n_samples, n_samples)
            Similarity matrix defining the similarities between all paris of available samples, e.g., S[i,j] describes
            the similarity between the samples x_i and x_j.
            Default similarity matrix is the unit matrix.
        data_set_: base.DataSet
            Data set containing samples, class labels, and optionally confidences of annotator(s).
        random_state_: numeric | np.random.RandomState
            Random state for annotator selection.

        References
        ----------
        [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra Spiliopoulou.
            Multi-Class Probabilistic Active Learning,
            vol. 285 of Frontiers in Artificial Intelligence and Applications, pages 586-594. IOS Press, 2016
        [2] Georg Krempl, Daniel Kottke, Vincent Lemaire.
            Optimised probabilistic active learning (OPAL),
            vol. 100 oof Machine Learning, pages 449-476. Springer, 2015
        """
        super().__init__(data_set=kwargs.pop('data_set', None), **kwargs)

        self.n_classes_ = kwargs.pop('n_classes', None)
        if not isinstance(self.n_classes_, int) or self.n_classes_ < 2:
            raise TypeError(
                "n_classes must be an integer and at least 2"
            )

        self.S_ = check_array(kwargs.pop('S', np.eye(len(self.data_set_))))
        if np.size(self.S_, axis=0) != np.size(self.S_, axis=1) or np.size(self.S_, axis=0) != len(self.data_set_):
            raise ValueError(
                "S must be a squared matrix where the number of rows is equal to the number of samples"
            )

        self.m_max_ = int(kwargs.pop('m_max', 1))
        if self.m_max_ < 1 or self.m_max_ > self.data_set_.n_annotators_:
            raise ValueError("'m_max' must be in the set {1, ..., n_annotators}")

        self.alpha_x_ = float(kwargs.pop('alpha_x', 1))
        self.alpha_c_ = float(kwargs.pop('alpha_c', 1))

        self.bam_ = kwargs.pop('bam', None)
        if not isinstance(self.bam_, BAM):
            raise TypeError("'bam' must be an instance of src.models.BAM")

    def compute_scores(self):
        """Compute score for each unlabeled sample. Score is to be maximized.

        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled sample.
        """
        is_unlabeled = np.isnan(self.data_set_.y_)
        unlabeled_indices = self.data_set_.get_unlabeled_indices()
        n_labels_per_annotator = self.data_set_.len_labeled(per_annotator=True)
        scores = np.full((len(self.data_set_), self.data_set_.n_annotators_), np.nan)

        # fit beta annotators model
        self.bam_.fit(self.data_set_.X_, self.data_set_.y_)

        # estimate mean annotation performances
        A_mean = self.bam_.predict_proba(self.data_set_.X_)[:, :, 0].T
        print('mean annotation performance values: {}'.format(np.mean(A_mean, axis=0)))
        self.data_set_.c_ = A_mean

        # sample annotation performances
        A_sample = A_mean.copy()
        A_rand = self.random_state_.random(A_sample.shape)
        if np.sum(n_labels_per_annotator <= 0):
            A_sample += 1 / (n_labels_per_annotator + 1) * 100
        A_sample[~is_unlabeled] = -1
        annotator_indices = np.lexsort((A_rand, -A_sample), axis=1)[:, :self.m_max_]

        # mask annotation performance values
        A = np.array([A_mean[np.arange(len(self.data_set_)), annotator_indices[:, a]] for a in range(self.m_max_)]).T

        # compute frequency estimates for evaluation set (K_x)
        Z = compute_vote_vectors(y=self.data_set_.y_, c=self.data_set_.c_, n_unique_votes=self.n_classes_,
                                 probabilistic=False)
        K_x = self.S_ @ Z

        # compute frequency estimates for candidate set and each annotator
        K_c = np.empty((self.data_set_.n_annotators_, len(self.data_set_), self.n_classes_))
        for a in range(self.data_set_.n_annotators_):
            Z_a = compute_vote_vectors(y=self.data_set_.y_[:, a], n_unique_votes=self.n_classes_)
            K_c[a] = (self.S_ @ Z_a)

        A = A[unlabeled_indices]
        annotator_indices = annotator_indices[unlabeled_indices]
        S = self.S_[unlabeled_indices]
        K_c = np.array([K_c[annotator_indices[:, a], unlabeled_indices] for a in range(self.m_max_)])
        scores[unlabeled_indices, annotator_indices[:, 0]] = xpal_gain(K_c=K_c, K_x=K_x, S=S, A=A,
                                                                       alpha_x=self.alpha_x_, alpha_c=self.alpha_c_)

        return scores


def xpal_gain(K_c, K_x=None, S=None, A=None, alpha_x=1, alpha_c=1):
    """
    Computes the expected probabilistic gain.

    Parameters
    ----------
    K_c: array-like, shape (n_candidate_samples, n_classes)
        Kernel frequency estimate vectors of the candidate samples.
    K_x: array-like, shape (n_evaluation_samples, n_classes), optional (default=K_c))
        Kernel frequency estimate vectors of the evaluation samples.
    S: array-like, shape (n_candidate_samples, n_evaluation_samples), optional (default=np.eye(n_candidate_samples))
        Similarities between all pairs of candidate and evaluation samples
    alpha_x: array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    alpha_c: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.

    Returns
    -------
    gains: numpy.ndarray, shape (n_candidate_samples)
        Computed expected gain for each candidate sample.
    """
    # check kernel frequency estimates of candidate samples
    n_annotators = K_c.shape[0]
    n_candidate_samples = K_c.shape[1]
    n_classes = K_c.shape[2]

    # check kernel frequency estimates of evaluation samples
    K_x = K_c if K_x is None else check_array(K_x)
    n_evaluation_samples = K_x.shape[0]
    if n_classes != K_x.shape[1]:
        raise ValueError("'K_x' and 'K_c' must have one column per class")

    # check similarity matrix
    S = np.eye(n_candidate_samples) if S is None else check_array(S)
    if S.shape[0] != n_candidate_samples or S.shape[1] != n_evaluation_samples:
        raise ValueError("'S' must have the shape (n_candidate_samples, n_evaluation_samples)")

    # check label accuracies
    A = np.ones(n_candidate_samples, n_annotators) if A is None else check_array(A)
    if A.shape[0] != n_candidate_samples or A.shape[1] != n_annotators:
        raise ValueError("'A' must have the shape (n_candidate_samples, n_annotators)")

    # check prior parameters
    if hasattr(alpha_c, "__len__") and len(alpha_c) != n_classes:
        raise ValueError("'alpha_c' must be either a float > 0 or array-like with shape (n_classes)")
    if hasattr(alpha_x, "__len__") and len(alpha_x) != n_classes:
        raise ValueError("'alpha_x' must be either a float > 0 or array-like with shape (n_classes)")

    # uniform risk matrix
    R = 1 - np.eye(n_classes)

    # compute possible risk differences
    class_vector = np.arange(n_classes, dtype=int)
    R_diff = np.array([[R[:, y_hat] - R[:, y_hat_l] for y_hat_l in class_vector] for y_hat in class_vector])

    # compute current error per evaluation sample and class
    R_x = K_x @ R

    # compute current predictions
    y_hat = np.argmin(R_x, axis=1)

    # compute required labels per class to flip decision
    with np.errstate(divide='ignore', invalid='ignore'):
        D_x = np.nanmin(np.divide(R_x - np.min(R_x, axis=1, keepdims=True), R[:, y_hat].T), axis=1)
        D_x = np.tile(D_x, (len(S), 1))

    # indicates where a decision flip can be reached
    A_max = np.sum(A, axis=1, keepdims=True)
    I = D_x - A_max * S < 0
    print('#decision_flips: {}'.format(np.sum(I)))

    # compute normalization constants per candidate sample
    K_c_alpha_c_norm = K_c + alpha_c
    K_c_alpha_c_norm /= K_c_alpha_c_norm.sum(axis=2, keepdims=True)

    # stores gain per candidate sample
    gains = np.zeros((n_candidate_samples, n_annotators))

    # compute gain for each candidate sample
    flip_indices = np.argwhere(np.sum(I, axis=1) > 0)[:, 0]
    for ik_c in flip_indices:
        for m in range(n_annotators):
            for class_idx in range(n_classes):
                norm = K_c_alpha_c_norm[:m + 1, ik_c, class_idx].prod()
                l_vec = np.zeros(n_classes)
                l_vec[class_idx] = A[ik_c, :m + 1].sum()
                K_l = (S[ik_c, I[ik_c]] * l_vec[:, np.newaxis]).T
                K_new = K_x[I[ik_c]] + K_l
                y_hat_l = np.argmin(K_new @ R, axis=1)
                K_new += alpha_x
                K_new /= np.sum(K_new, axis=1, keepdims=True)
                gains[ik_c, m] += norm * np.sum(K_new * R_diff[y_hat[I[ik_c]], y_hat_l])

    # compute average gains over evaluation samples
    gains /= n_evaluation_samples
    gains /= np.arange(1, n_annotators + 1).reshape(1, -1)
    print(np.unravel_index(gains.argmax(), gains.shape))

    return np.max(gains, axis=1)
