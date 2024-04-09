from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import ott.geometry.costs as costs
from numpy.typing import ArrayLike
from ott.geometry import costs, geometry
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import sinkhorn

from utils.ot_cost_fns import create_cost_matrix, get_nearest_neighbors


def sinkhorn_matching(
    X: ArrayLike,
    Y: ArrayLike,
    tau_a: float = 1,
    tau_b: float = 1,
    epsilon: float = 1e-2,
    scale_cost: Any = "mean",
    cost_fn=costs.SqEuclidean(),
    geometry: Literal["pointcloud", "graph", "geodesic"] = "pointcloud",
):
    B = X.shape[0]
    if geometry == "pointcloud":
        geom = PointCloud(
            jnp.reshape(X, [B, -1]),
            jnp.reshape(Y, [B, -1]),
            epsilon=epsilon,
            scale_cost=scale_cost,
            cost_fn=cost_fn,
        )
    else:
        # ! Notice we put here double the neighbors. Because we are "duplicate X" [X,X] in cost matrix, one option would be not to give X (which is how this is originally intended)
        # But this is just for testing / quick visualizing & We don't want to change that many parts of the code for it.
        cm = create_cost_matrix(
            X=X,
            Y=Y,
            k_neighbors=60,
            cost_fn=cost_fn,
            geometry=geometry,
        )
        geom = geometry.Geometry(
            cost_matrix=cm,
            epsilon=epsilon,
            scale_cost=scale_cost,
        )

    ot_out = sinkhorn.solve(
        geom,
        tau_a=tau_a,
        tau_b=tau_b,
    )
    return ot_out.matrix


def distance_matching(f, X, Y=None, normalize: bool = True):
    if Y is None:
        matrix = jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(X))(X)
    else:
        matrix = jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(Y))(X)

    if normalize:
        matrix /= jnp.sum(matrix)
    return matrix


def rowwise_keep_top_k(matrix: ArrayLike, top_k: int):
    top_k_matrix = np.zeros_like(matrix)
    for i, row in enumerate(matrix):
        indices = np.argsort(row)[-top_k:]
        top_k_matrix[i, indices] = matrix[i, indices]
    return top_k_matrix


def rowwise_take_top_k(
    matrix: ArrayLike, top_k: int, labels: Optional[ArrayLike] = None
):
    """Extracts top_k values row-wise from matrix.

    Args:
        matrix (np.ndarray): distance matrix
        top_k (int): Number of top elements to take
        labels (Optional[np.ndarray], optional): labels from which to extract elements, if None defaults to extract from matrix. Defaults to None.

    Returns:
        np.ndarray: Submatrix with corresponding elements row-wise
    """
    mask = np.argsort(-matrix, axis=1)[:, :top_k]

    if labels is None:
        top_k_labels = matrix[np.arange(matrix.shape[0])[:, None], mask]
    else:
        top_k_labels = labels[mask]
    return top_k_labels
