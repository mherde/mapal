import numpy as np
import os.path
import pandas as pd

from itertools import compress

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score


def load_data(data_set_name):
    """
    Loads data set of given data set name.

    Parameters
    ----------
    data_set_name: str
        Name of the data set.

    Returns
    -------
    X: array-like, shape (n_samples, n_features)
        Samples as feature vectors.
    y_true: array-like, shape (n_samples)
        True class labels of samples.
    y: array-like, shape (n_samples, n_annotators_)
        Class label of each annotator (only available for grid data set).
    """
    try:
        # look locally for data
        abs_path = os.path.abspath(os.path.dirname(__file__))
        relative_path = '../../data/' + data_set_name + '.csv'
        data_set = pd.read_csv(os.path.join(abs_path, relative_path))
        columns = list(data_set.columns.values)
        features = list(compress(columns, [c.startswith('x_') for c in columns]))
        labels = list(compress(columns, [c.startswith('y_') for c in columns]))

        # Get features.
        X = np.array(data_set[features], dtype=np.float64)

        # Getting assumed true labels.
        y_true = np.array(data_set['y'].values, dtype=int)

        # Get labels of annotators.
        y = np.array(data_set[labels], dtype=int)

        # Ensure label reach from 0 to n_classes-1.
        le = LabelEncoder().fit(np.unique(np.column_stack((y, y_true))))
        y_true = le.transform(y_true)
        y = np.column_stack([le.transform(y[:, a]) for a in range(y.shape[1])])

    except FileNotFoundError:
        relative_path = '../../data/data_set_ids.csv'
        data_set = pd.read_csv(os.path.join(abs_path, relative_path))
        idx = data_set[data_set['name'] == data_set_name].index.values.astype(int)[0]
        data_set = fetch_openml(data_id=data_set.at[idx, 'id'])
        X = data_set.data
        y_true = data_set.target
        le = LabelEncoder().fit(y_true)
        y_true = le.transform(y_true)
        y = None

    return X, y_true, y


def investigate_data_set(data_set_name):
    X, y_true, y = load_data(data_set_name)
    n_instances_per_class = np.unique(y_true, return_counts=True)[1]
    n_features = X.shape[1]
    annotation_perfs = np.array([accuracy_score(y_pred=y[:, a], y_true=y_true) for a in range(y.shape[1])])
    return n_features, n_instances_per_class, annotation_perfs


def preprocess_2d_data_set(data_set_name, res=101):
    X, y_true, y = load_data(data_set_name=data_set_name)
    X = StandardScaler().fit_transform(X)
    x_1_vec = np.linspace(min(X[:, 0])-0.5, max(X[:, 0])+0.5, res)
    x_2_vec = np.linspace(min(X[:, 1])-0.5, max(X[:, 1])+0.5, res)
    X_1_mesh, X_2_mesh = np.meshgrid(x_1_vec, x_2_vec)
    mesh_instances = np.array([X_1_mesh.reshape(-1), X_2_mesh.reshape(-1)]).T
    n_samples = len(X)
    n_features = np.size(X, 1)
    nominator = 2 * n_samples * n_features
    denominator = (n_samples - 1) * np.log((n_samples - 1) / ((np.sqrt(2) * 10 ** -6) ** 2))
    bandwidth = np.sqrt(nominator / denominator)
    gamma = 0.5 * (bandwidth ** (-2))
    return X, y_true, y, X_1_mesh, X_2_mesh, mesh_instances, gamma


class Mixture:
    """
    Represents a Mixture of distributions.

    Parameters
    ----------
    priors: array-like, shape=[n_distributions]
        Prior probabilities for the given distributions.
    base_dists: array-like, shape=[n_distributions]
        Underlying distributions.
    classes: array-like, shape=[n_distributions]
        Class label of each distribution.
    """
    def __init__(self, priors, base_dists, classes=None):
        self.priors = priors
        self.base_dists = base_dists
        if classes is None:
            classes = [None] * len(priors)
        self.classes = classes
        self.n_dists = len(self.priors)
        self.n_classes = len(np.unique(self.classes))

    def rvs(self, size):
        """Random variates of given type.

        Parameters
        ----------
        size: array-like, shape=[n_samples, n_features]
            Sizes of the resulting data set.

        Returns
        -------
        X: array-like, shape=[n_samples, n_features]
            Dataset with samples as feature vectors.
        Y: array-like, shape=[n_samples]
            Class label of each sample.
        """
        random_state = np.random.RandomState(42)
        n_inst_per_base_dists = random_state.multinomial(size[0], self.priors)

        X = list()
        Y = list()
        for i, n_inst_per_base_dist in enumerate(n_inst_per_base_dists):
            X.append(self.base_dists[i].rvs([n_inst_per_base_dist, *size[1:]]))
            Y.append(np.ones((n_inst_per_base_dist, 1)) * self.classes[i])
        resort = random_state.permutation(size[0])
        X = np.vstack(X)[resort]
        Y = np.array(np.vstack(Y)[resort].ravel(), int)
        return X, Y

    def pdf(self, x, c=None):
        """Probability density function at x of the given RV.

        Parameters
        ----------
        x: array-like, shape=[n_samples, n_features]
            Sample to evaluate pdf.
        c: array-like, shape=[n_samples]
            Class labels.

        Returns
        -------
        densities: array-like, shape=[n_samples]
            Density of a sample, if it belongs to class c.
        """
        if c is None:
            c = list(np.unique(self.classes))
        if type(c) is not list:
            c = [c]
        c_idx = np.where([self.classes[i] in c for i in range(self.n_dists)])[0]
        return np.sum([self.priors[i] * self.base_dists[i].pdf(x)
                       for i in c_idx], axis=0)
