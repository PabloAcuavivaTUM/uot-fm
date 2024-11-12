from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import logging 

import dm_pix as pix
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from ott.geometry.pointcloud import geometry, PointCloud
from ott.solvers.linear import sinkhorn

from .ot_cost_fns import dist_fns, cost_fns, create_cost_matrix, fmatching
from .miscellaneous import EasyDict


@dataclass
class GenerationSampler:
    """Data sampler for a generation task with optional weighting."""

    data: jax.Array
    batch_size: int
    weighting: Optional[jnp.ndarray] = None
    do_flip: bool = True

    def __post_init__(self):
        # Weighting needs to have the same length as data.
        if self.weighting is not None:
            assert self.data.shape[0] == self.weighting.shape[0]

        @eqx.filter_jit
        def _sample(key: jax.random.KeyArray) -> jax.Array:
            """Jitted sample function."""
            x = jax.random.choice(
                key, self.data, shape=[self.batch_size], p=self.weighting
            )
            x = x / 127.5 - 1.0
            if self.do_flip:
                x = pix.random_flip_left_right(key, x)
            x = jnp.transpose(x, [0, 3, 1, 2])
            return jr.normal(key, shape=x.shape), x

        self.sample = _sample

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        """Sample data."""
        return self.sample(key)


@dataclass
class BatchResampler:
    """Batch resampler based on (Unbalanced) Optimal Transport."""

    batch_size: int
    batch_size_matching : int 
    tau_a: float = 1.0
    tau_b: float = 1.0
    epsilon: float = 1e-2
    cost_fn: str = "sqeuclidean"
    geometry: Literal["pointcloud", "graph", "geodesic"] = "pointcloud"
    geometry_cost_matrix_kwargs : Optional[dict] = None
    matching_method : Literal["ot", "softmax_dist", "abs_dist"] = "ot"
    compare_on : Literal["data", "embedding"] = "data"
    pointcloud_batch_size : Optional[int] = None

    def __post_init__(self):
        if self.geometry_cost_matrix_kwargs is None:
            self.geometry_cost_matrix_kwargs = dict()

        @eqx.filter_jit(donate="all")
        def _resample(
            key: jr.KeyArray,
            source_batch: EasyDict,
            target_batch: EasyDict,
        ) -> Tuple[jax.Array, jax.Array]:
            """Jitted resample function."""
            # solve regularized ot between batch_source and batch_target reshaped to (batch_size, dimension)
            # Change source_batch & target_batch for EasyDict
            # source_batch[self.compare_on], target_batch[self.compare_on]
            if self.matching_method == "ot":                
                if self.geometry == "pointcloud":
                    geom = PointCloud(
                        jnp.reshape(source_batch[self.compare_on], [self.batch_size_matching, -1]),
                        jnp.reshape(target_batch[self.compare_on], [self.batch_size_matching, -1]),
                        epsilon=self.epsilon,
                        scale_cost="mean",
                        cost_fn=cost_fns[self.cost_fn],
                        batch_size=self.pointcloud_batch_size,
                    )
                else: # Create a cost matrix
                    
                    if self.geometry == 'pointcloud_combine': # Combine multiple point clouds for a single cost matrix 
                        
                        combine0 = self.geometry_cost_matrix_kwargs['combine0']
                        combine1 = self.geometry_cost_matrix_kwargs['combine1']

                        cost_fn0 = self.geometry_cost_matrix_kwargs['cost_fn0']
                        cost_fn1 = self.geometry_cost_matrix_kwargs['cost_fn1']
                        
                        cm0 = PointCloud(
                            jnp.reshape(source_batch[combine0], [self.batch_size_matching, -1]),
                            jnp.reshape(target_batch[combine0], [self.batch_size_matching, -1]),
                            epsilon=self.epsilon,
                            scale_cost="mean",
                            cost_fn=cost_fns[cost_fn0],
                            batch_size=self.pointcloud_batch_size,
                        ).cost_matrix

                        cm1 = PointCloud(
                            jnp.reshape(source_batch[combine1], [self.batch_size_matching, -1]),
                            jnp.reshape(target_batch[combine1], [self.batch_size_matching, -1]),
                            epsilon=self.epsilon,
                            scale_cost="mean",
                            cost_fn=cost_fns[cost_fn1],
                            batch_size=self.pointcloud_batch_size,
                        ).cost_matrix
                        
                        beta = self.geometry_cost_matrix_kwargs['combine_beta']
                        
                        cm = beta*cm0 + (1-beta)*cm1

                    else: # Geodesic cost matrix                    
                        cm = create_cost_matrix(
                            X=jnp.reshape(source_batch[self.compare_on], [self.batch_size_matching, -1]),
                            Y=jnp.reshape(target_batch[self.compare_on], [self.batch_size_matching, -1]),
                            k_neighbors=30,
                            cost_fn=cost_fns[self.cost_fn],
                            geometry=self.geometry,
                            **self.geometry_cost_matrix_kwargs
                        )

                    geom = geometry.Geometry(
                        cost_matrix=cm,
                        epsilon=self.epsilon,
                        scale_cost="mean",
                    )

                ot_out = sinkhorn.solve(geom, tau_a=self.tau_a, tau_b=self.tau_b)
                transition_matrix = ot_out.matrix

            elif self.matching_method in ["softmax_dist", "abs_dist"]:
                transition_matrix = fmatching(
                    dist_fns[self.cost_fn], 
                    X=source_batch, 
                    Y=target_batch, 
                    softmax=self.matching_method == "softmax_dist", 
                    dist_mult=100, 
                    top_k=4, 
                    as_coupling=True,
                )
            else:
                raise ValueError(f'Invalid matching_method provided {self.matching_method}.')

            # get flattened log transition matrix
            transition_matrix = jnp.log(transition_matrix.flatten())

            # sample from transition_matrix
            indeces = jax.random.categorical(
                key, transition_matrix, shape=[self.batch_size]
            )
            
            resampled_indeces_source = indeces // self.batch_size_matching
            resampled_indeces_target = indeces % self.batch_size_matching

            return resampled_indeces_source, resampled_indeces_target 

            
        self.resample = _resample

    def __call__(
        self,
        key: jr.KeyArray,
        source_batch: EasyDict,
        target_batch: EasyDict,
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample data."""
        
        resampled_indeces_source, resampled_indeces_target = self.resample(
            key, source_batch, target_batch
        )

        return (
                source_batch.slice(resampled_indeces_source),
                target_batch.slice(resampled_indeces_target),
            )