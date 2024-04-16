import jax
import jax.numpy as jnp
import ott.geometry.costs as costs


# CUSTOM COSTS
@jax.tree_util.register_pytree_node_class
class CoulombCost(costs.CostFn):
    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return 1.0 / (jnp.linalg.norm((x - y), ord=2, axis=-1))


# [WARNING]: Computing the histogram is relatively slow. Therefore is probabibly better to consider it as an embedding (Same we would do with CLIP).
@jax.tree_util.register_pytree_node_class
class HistCost(costs.CostFn):
    def __init__(self, nchannels: int = 1):
        super().__init__()
        self.nchannels = nchannels

    def pairwise(self, x: jax.Array, y: jax.Array) -> float:
        def compute_histogram(channel):
            return jnp.histogram(channel, bins=256)[0]

        x = x.reshape(self.nchannels, -1)
        y = y.reshape(self.nchannels, -1)

        histx = jnp.stack(jax.vmap(compute_histogram)(x)).ravel()
        histy = jnp.stack(jax.vmap(compute_histogram)(y)).ravel()

        return jnp.linalg.norm((histx - histy), ord=2, axis=-1)


# ---
cost_fns = dict(
    sqeuclidean=costs.SqEuclidean(),
    l1=costs.PNormP(p=1),
    euclidean=costs.Euclidean(),
    arcos=costs.Arccos(n=1),
    cosine=costs.Cosine(),
    coulomb=CoulombCost(),
    elastic_l1=costs.ElasticL1(),
    elastic_l2=costs.ElasticL2(),
    elastic_stvs=costs.ElasticSTVS(),
)

#####################################
#   Graphs & Geodesics functionality
#####################################
from typing import Any, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import ott.geometry.costs as costs
from ott.geometry import geodesic, graph, pointcloud


def get_nearest_neighbors(
    X: jax.Array,
    Y: Optional[jax.Array] = None,
    k_neighbors: int = 30,
    cost_fn: Optional[costs.CostFn] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    concat = jnp.concatenate((X, Y), axis=0) if Y is not None else X
    pairwise_distances = pointcloud.PointCloud(
        concat,
        concat,
        cost_fn=cost_fn,
    ).cost_matrix

    distances, indices = jax.lax.approx_min_k(
        pairwise_distances,
        k=k_neighbors,
        recall_target=0.95,
        aggregate_to_topk=True,
    )

    return distances, indices


def create_cost_matrix(
    X: jax.Array,
    Y: Optional[jax.Array] = None,
    k_neighbors: int = 30,
    cost_fn: Optional[costs.CostFn] = None,
    geometry: Literal["graph", "geodesic"] = "geodesic",
    **kwargs: Any,
) -> jnp.array:
    distances, indices = get_nearest_neighbors(
        X=X,
        Y=Y,
        k_neighbors=k_neighbors,
        cost_fn=cost_fn,
    )

    # Increase weight in adjacency matrix of close points. Make further points with smaller value
    distances = jnp.exp(-distances)

    n = len(X) + len(Y) if Y is not None else len(X)
    a = jnp.zeros((n, n))
    adj_matrix = a.at[
        jnp.repeat(jnp.arange(n), repeats=k_neighbors).flatten(),
        indices.flatten(),
    ].set(distances.flatten())

    geometry_fn = (
        geodesic.Geodesic.from_graph
        if geometry == "geodesic"
        else graph.Graph.from_graph
    )
    # default t=1e-3, single cell data: 100. Try different ts: [0.1, 1, 10, 100]

    if Y is not None:
        cm = geometry_fn(
            adj_matrix,
            normalize=kwargs.pop("normalize", True),
            **kwargs,
        ).cost_matrix[: len(X), len(X) :]
    else:
        cm = geometry_fn(
            adj_matrix,
            normalize=kwargs.pop("normalize", True),
            **kwargs,
        ).cost_matrix

    return cm
