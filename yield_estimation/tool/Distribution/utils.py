import numpy as np

def normalize(x):
    """
        normalize vector to a unit vector
    """
    len = (x*x).sum(axis=-1)[:,None] ** 0.5
    return x / len

def cos_distance(x, v):
    """
    :param x: shape: N1 x d, each raw represent a sample denoted as x1, x2, x3 ... xN1
    :param v: shape: N2 x d, each raw represent a sample denoted as v1, v2, v3 ... vN1
    :return:
        cosin_distance_matrix: shape N1 x N2,
        element in raw i column j presents the cosine distance between xi and vj
    """

    x = x.reshape(-1, x.shape[-1])
    v = v.reshape(-1, v.shape[-1])

    x_sum = (x * x).sum(axis=-1).reshape([-1,1]) ** 0.5
    v_sum = (v * v).sum(axis=-1).reshape([-1,1]) ** 0.5

    cosine_distance_matrix = 1 - np.dot(x, v.T) / np.dot(x_sum, v_sum.T)

    if (len(cosine_distance_matrix.shape) == 2 and cosine_distance_matrix.shape[-1] == 1):
        cosine_distance_matrix = float(cosine_distance_matrix)

    return cosine_distance_matrix

