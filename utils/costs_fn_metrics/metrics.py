import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from ott.geometry import geometry
from ott.solvers import linear


def similarity_top_k(
    label: ArrayLike, top_k_labels: ArrayLike, mask: ArrayLike = None
) -> float:
    # Distance function between two labels
    if mask is None:
        diff_fn = jax.vmap(
            lambda x, y: 1
            - jnp.sum(jnp.abs(x - y), axis=1) / (jnp.sum(x) + jnp.sum(y, axis=1)),
            in_axes=(0, 0),
        )
    else:
        diff_fn = jax.vmap(
            lambda x, y: 1
            - jnp.sum(jnp.abs(x[mask] - y[:, mask]), axis=1)
            / (jnp.sum(x[mask]) + jnp.sum(y[:, mask], axis=1)),
            in_axes=(0, 0),
        )
    # ---
    similarity = diff_fn(label, top_k_labels)  # [B, k]

    similarity = jnp.mean(
        similarity, axis=1
    )  # [B,] Aggregate top_k per element, in this case uniform weighting

    similarity = jnp.mean(similarity, axis=0)  # Combine the score for all elements
    return similarity


def discrete_wasserstein_distance(
    a: ArrayLike,
    b: ArrayLike,
    cost_matrix: Optional[ArrayLike] = None,
) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Only similar length arrays may be comparend! len(a)={len(a)}!={len(b)}=len(b)"
        )

    if cost_matrix is None:
        cost_matrix = np.ones(shape=(len(a), len(a))) - np.eye(len(a))

    geom = geometry.Geometry(cost_matrix=cost_matrix)

    sol = linear.solve(
        geom,
        a=a,
        b=b,
    )

    return sol.reg_ot_cost


def rowwise_correlation(matrix0: ArrayLike, matrix1: ArrayLike):
    centered_matrix0 = matrix0 - np.mean(matrix0, axis=1, keepdims=True)
    centered_matrix1 = matrix1 - np.mean(matrix1, axis=1, keepdims=True)

    corr = np.sum(centered_matrix0 * centered_matrix1, axis=1) / (
        np.sqrt(np.sum(centered_matrix0**2, axis=1))
        * np.sqrt(np.sum(centered_matrix1**2, axis=1))
    )
    return corr
