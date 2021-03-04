import numpy as np
from scipy import spatial
import torch
from matplotlib import pyplot as plt


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def rbf_kernel(xa, xb):
    """Exponentiated quadratic with σ=1"""
    # L2 distance (Squared Euclidian)
    dist = -2*spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(dist)


def shuffle_along_axis_and_filter(a, axis, points):
    idx = torch.randperm(a.shape[axis])
    return a[:, idx[:points], :]


def sample_gaussian(number_points, number_fct=1, sigma_val=None, mean_val=None):
    X = np.expand_dims(np.linspace(-2, 2, number_points), 1)

    if sigma_val is None:
        sigma_val = rbf_kernel(X, X)
    if mean_val is None:
        mean_val = np.zeros(len(sigma_val))

    functions = np.random.multivariate_normal(mean=mean_val, cov=sigma_val, size=number_fct)

    return functions, X


def get_training_data(number_points, batch_size, context_points, plot=False):
    # Generate one function
    curve, xs = sample_gaussian(number_points)
    curve = np.around(curve, 2)
    data = np.concatenate([xs, curve.T], axis=1)
    idx = np.random.choice(data.shape[0], (batch_size, number_points), replace=True)
    sh_data_c = data[idx]

    sh_data = np.tile(data, (batch_size, 1, 1))

    # sh_data_c = shuffle_along_axis_and_filter(sh_data, axis=1, points=context_points)
    c_points = sh_data_c[:, :context_points, :]
    t_points = sh_data

    t_points_X = np.split(t_points, 2, axis=2)[0]
    t_points_y = np.split(t_points, 2, axis=2)[1]

    if plot:
        plt.plot(xs, curve.T)
        plt.scatter(c_points[0].T[0], c_points[0].T[1])
        plt.legend()
        plt.show()
        plt.plot(xs, curve.T)
        plt.scatter(c_points[1].T[0], c_points[1].T[1])
        plt.legend()
        plt.show()

    return c_points, t_points_X, t_points_y, data
