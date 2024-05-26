import jax
import jax.numpy as jnp
import ott.geometry.costs as costs
import logging 

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

dist_fns = dict(
    sqeuclidean=lambda x,y: jax.numpy.linalg.norm(((x-y).ravel()), ord=2),
    l1=lambda x,y: jax.numpy.linalg.norm((x-y.ravel()), ord=1),
    euclidean=lambda x,y: jax.numpy.linalg.norm(((x-y).ravel()), ord=2)**2,
    dot=lambda x,y: jnp.dot(x,y), # Cosine if vectors are normalized
    cosine=lambda x,y: jnp.dot(x.ravel(),y.ravel()) / (jax.numpy.linalg.norm((y.ravel()), ord=2)*jax.numpy.linalg.norm((x.ravel()), ord=2)),
    coulomb=lambda x,y: 1 / jnp.max(1e-9, jax.numpy.linalg.norm((x-y).ravel(), ord=2)),
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
    if geometry not in ["graph", "geodesic"]:
        raise ValueError(f'geodesic must be in ["graph", "geodesic"], given {geodesic}')
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


##########################################################################
# It is not really related to OT transport but we leave it here for now
def take_top_k_rowwise(matrix : jax.Array, top_k : int = 3):
    top_indices = jnp.argsort(matrix, axis=1)[:, -top_k:]
    mask = jnp.zeros_like(matrix, dtype=bool)
    rows = jnp.arange(matrix.shape[0])[:, None]
    mask = mask.at[rows, top_indices].set(True)

    return jnp.where(mask, matrix, jnp.zeros_like(matrix))

def fmatching(f, X, Y=None,  
                      softmax : bool = True, 
                      dist_mult : int = 1,
                      as_coupling: bool =True, 
                      top_k : int = None):
    if Y is None:
        matrix = jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(X))(X)
    else:
        matrix = jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(Y))(X)
    
    if Y is None:
        # Set self-distance-similarity to 0
        matrix = matrix.at[jnp.arange(matrix.shape[0]), jnp.arange(matrix.shape[0])].set(0)

    
    if softmax:
        # Notice here we are making further away (in distance metrics) have higher score
        # This is to make is similar to how CLIP works (high cosine similarity score for related points)
        matrix = jax.nn.softmax(matrix * dist_mult, axis=1)  

    else:
        # Closer points should have a higher value 
        m = jnp.max(matrix)
        matrix = (m - matrix) / m
        matrix = jnp.exp(matrix * dist_mult)
    
    
    if top_k is not None:
        matrix = take_top_k_rowwise(matrix, top_k=top_k)
        # matrix = take_top_k_rowwise(matrix.T, top_k=top_k).T

    if as_coupling: # Normalize to a coupling
        matrix /= jnp.sum(matrix)

    return matrix