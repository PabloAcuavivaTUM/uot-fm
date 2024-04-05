import jax
import jax.numpy as jnp
import ott.geometry.costs as costs

__all__ = ["ot_cost_fns"]


# CUSTOM COSTS
@jax.tree_util.register_pytree_node_class
class CoulombCost(costs.CostFn):
    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return 1.0 / (jnp.linalg.norm((x - y), ord=2, axis=-1))


# ---
ot_cost_fns = dict(
    sqeuclidean=costs.SqEuclidean(),
    l1_cost=costs.PNormP(p=1),
    euclidean=costs.Euclidean(),
    arcos=costs.Arccos(n=1),
    cosine=costs.Cosine(),
    coulomb=CoulombCost(),
)
