from typing import Callable, Optional

import jax
import jax.numpy as jnp


def distance_matrix(
    X: jax.Array,
    Y: Optional[jax.Array] = None,
    dist: Callable = lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2)),
):
    if Y is None:
        Y = X
    return jax.vmap(lambda x: jax.vmap(lambda y: dist(x, y))(Y))(X)


# Based on paper "Improved Precision and Recall Metric for Assessing Generative Models"
def estimate_precision(x_real: jax.Array, x_generated: jax.Array, k: int = 5):
    return manifold_estimate(x_real, x_generated, k=k)


def estimate_recall(x_real: jax.Array, x_generated: jax.Array, k: int = 5):
    return manifold_estimate(x_generated, x_real, k=k)


def manifold_estimate(sigma_a: jax.Array, sigma_b: jax.Array, k: int = 5):
    distances_a = distance_matrix(X=sigma_a)  # Pairwise distance sigma_a
    r = jnp.sort(distances_a, axis=1)[:, k]  # (k+1)-th smallets value

    distances_ba = distance_matrix(X=sigma_b, Y=sigma_a)

    # Compute how many points from sigma_b are within the approximated manifold of sigma_a
    n = jnp.sum(jnp.any((distances_ba - r[None, :]) < 0, axis=1))

    return n / sigma_b.shape[0]


def calculate_same_class_perc(
    x_real: jax.Array,
    x_generated: jax.Array,
    top_k: int = 5,
) -> float:

    X = jnp.concatenate((x_real, x_generated))
    N_real = len(x_real)
    distances = distance_matrix(X)[:N_real]
    closest_indices = jnp.argsort(distances, axis=1)[:, 1 : top_k + 1]

    return jnp.mean(closest_indices < N_real)
