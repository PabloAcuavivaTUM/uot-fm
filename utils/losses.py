import functools as ft
from typing import Any, Callable, Dict, List, Optional, Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from ml_collections import ConfigDict
from .miscellaneous import EasyDict
from functools import reduce, partial
from copy import deepcopy

# Two low dimensional matrices AA^T
# Fix rn_positions  

from typing import Optional 


# rn_positions = jr.choice(key=key_choice, a=jnp.arange(independent_terms), shape=(n,))

# N(0, SIGMA = AA^T))

# N(0, I) -> SVD -> Taking biggest singular values ->

def get_correlated_multivariate_normal_fn(n, independent_terms, key):
    rn_positions = jr.choice(key=key, a=jnp.arange(independent_terms), shape=(n,))
    def correlated_multivariate_normal(key : jr.KeyArray, x0_data : jax.Array, shape : Tuple[int, ...]):
        rn = jr.normal(key=key, shape=(independent_terms,))
        return rn[rn_positions].reshape(shape)
    
    return jax.jit(correlated_multivariate_normal, static_argnums=(1,))

def correlated_multivariate_normal_matrix_fn(key : jr.KeyArray, 
                                             x0_data : jax.Array, 
                                             shape : Tuple[int, ...], 
                                             rank : int = 100,
                                             ):
    m = shape[0]*shape[1]*shape[2] 
    A = jax.random.uniform(key, shape=(m, rank))
    # cov = jnp.matmul(A, A.T) 
    # cov = cov.at[jnp.diag_indices(m)].set(1)

    rn = jax.random.normal(key, shape=(rank,))
    
    return (A@rn).reshape(shape)



def _positional_encoding_2d(d_model, height, width):
    pe = jnp.zeros((height, width, d_model))
    
    y = jnp.arange(height, dtype=jnp.float32)
    x = jnp.arange(width, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y)
    xx = xx[:,:, None]  # Shape: (height, width, 1)
    yy = yy[:,:, None]  # Shape: (height, width, 1)
    
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    div_term = div_term[None, None, :]  # Shape: (1, 1, d_model // 2)

    ####
    pe.at[:,:, 0::2].set((jnp.sin(xx* div_term)*jnp.sin(yy* div_term)))
    pe.at[:,:, 1::2].set((jnp.cos(xx* div_term)*jnp.cos(yy* div_term)))
        
    return pe.transpose(2,0,1)



src_noises = dict(gaussian=lambda key, x0_data, shape: jr.normal(shape=shape, key=key),
                  gaussian01=lambda key, x0_data, shape: 0.1*jr.normal(shape=shape, key=key),
                  gaussian05=lambda key, x0_data, shape: 0.5*jr.normal(shape=shape, key=key),
                  gaussian1p2=lambda key, x0_data, shape: 1.2*jr.normal(shape=shape, key=key),
                  gaussian1p5=lambda key, x0_data, shape: 1.5*jr.normal(shape=shape, key=key),
                  gaussian2=lambda key, x0_data, shape: 2*jr.normal(shape=shape, key=key),
                    chisquare=lambda key, x0_data, shape: jr.chisquare(df=1, key=key, shape=shape) - 1,
                    uniform=lambda key, x0_data, shape: 2*(jr.uniform(key=key, shape=shape) - 0.5),  # Center [-1,1] 
                    uniform1p5=lambda key, x0_data, shape: 3*(jr.uniform(key=key, shape=shape) - 0.5),  # Center [-1.5,1.5]
                    uniform2=lambda key, x0_data, shape: 4*(jr.uniform(key=key, shape=shape) - 0.5),  # Center [-2,2] 
                    uniform2p5=lambda key, x0_data, shape: 5*(jr.uniform(key=key, shape=shape) - 0.5),  # Center [-2.5,2.5] 
                    exponential=lambda key, x0_data, shape: jr.exponential(key=key, shape=shape) - 1, 
                    beta33=lambda key, x0_data, shape: jr.beta(a=3,b=3, key=key, shape=shape) - 0.5, 
                    beta55=lambda key, x0_data, shape: jr.beta(a=5,b=5, key=key, shape=shape) - 0.5, 
                    beta27=lambda key, x0_data, shape: jr.beta(a=2,b=7, key=key, shape=shape)  - 2 / 7,
                    const_zero=lambda key, x0_data, shape: jnp.zeros_like(x0_data),
                    const_gaussian=lambda key, x0_data, shape: jr.normal(shape=shape, key=jr.PRNGKey(42)),
                    const_encoding=lambda key, x0_data, shape: _positional_encoding_2d(*shape), # TODO - REVISE
     )



def get_loss_builder(config: ConfigDict):
    if config.training.method == "flow":
        return FlowMatching(
            t1=config.t1,
            dt0=config.dt0,
            flow_sigma=config.training.flow_sigma,
            gamma=config.training.gamma,
            weight=lambda t: 1.0,
            solver=config.solver,
            is_genot=config.training.is_genot,
            genot=config.training.genot,
            key=jr.PRNGKey(config.seed),
        )
    elif config.training.method == "flow-vp-matching":
        raise NotImplementedError
    elif config.training.method == "flow-ve-matching":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown training method {config.training.method}")


def get_optimizer(config: ConfigDict):
    if config.optim.schedule == "constant":
        schedule = optax.constant_schedule(config.optim.learning_rate)
    elif config.optim.schedule == "linear":
        schedule = optax.linear_schedule(
            init_value=config.optim.learning_rate,
            end_value=1e-8,
            transition_steps=config.training.num_steps - config.optim.warmup,
        )
    elif config.optim.schedule == "polynomial":
        schedule = optax.polynomial_schedule(
            init_value=config.optim.learning_rate,
            end_value=1e-8,
            power=0.9,
            transition_steps=config.training.num_steps - config.optim.warmup,
        )
    elif config.optim.schedule == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=config.optim.learning_rate,
            decay_steps=config.training.num_steps - config.optim.warmup,
            alpha=1e-5 / config.optim.learning_rate,
        )
    else:
        raise ValueError(f"Unknown schedule type {config.optim.schedule}")
    if config.optim.warmup > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=1e-8,
            end_value=config.optim.learning_rate,
            transition_steps=config.optim.warmup,
        )
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, schedule],
            boundaries=[config.optim.warmup],
        )
    if config.optim.optimizer == "adam":
        optimizer = optax.adamw(
            learning_rate=schedule,
            b1=config.optim.beta_one,
            b2=config.optim.beta_two,
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.optimizer == "sgd":
        optimizer = optax.sgd(
            learning_rate=schedule,
            momentum=config.optim.momentum,
            nesterov=config.optim.nesterov,
        )
    elif config.optim.optimizer == "adabelief":
        optimizer = optax.adabelief(
            learning_rate=schedule,
            b1=config.optim.beta_one,
            b2=config.optim.beta_two,
            eps=config.optim.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer type {config.optim.type}")
    if config.optim.grad_clip > 0.0:
        optimizer = optax.chain(
            optimizer, optax.clip_by_global_norm(config.optim.grad_clip)
        )
    return optimizer


class FlowMatching:
    """Class for Flow Matching loss computation and sampling."""

    def __init__(
        self,
        t1: float,
        dt0: float,
        t0: float = 0.0,
        gamma: str = "constant",
        flow_sigma: Optional[float] = 0.1,
        weight: Optional[Callable[[float], float]] = lambda t: 1.0,
        solver: str = "tsit5",
        is_genot : bool = False,
        genot : Optional[ConfigDict] = None,  # Genot configuration 
        key = None,
    ):
        genot = genot or ConfigDict() # ! WARNING Will throw error when accessing .noise

        
        self.t1 = t1
        self.t0 = t0
        self.dt0 = dt0
        self.gamma = gamma
        self.sigma = flow_sigma
        self.weight = weight
        self.solver = solver
        self.is_genot = is_genot
        
        # Clean up a bit
        self.genot = deepcopy(genot)

        if self.genot:
            if 'x0_plus_' in self.genot.noise:
                self.genot.noise = self.genot.noise[8:] # Remove x0_plus
                self.x0_plus_noise = True
            else:
                self.x0_plus_noise = False
        
            if self.genot.noise in src_noises:
                self.noise_genot = src_noises[self.genot.noise]
            elif self.genot.noise == 'low_rank_gaussian':
                self.noise_genot = get_correlated_multivariate_normal_fn(self.genot.n, 
                                                                         self.genot.gaussian_independent_terms, 
                                                                         key,
                                                                         )
            elif self.genot.noise == 'low_rank_gaussian_matrix':
                self.noise_genot = partial(correlated_multivariate_normal_matrix_fn, 
                                       rank=self.genot.gaussian_independent_terms,
                                       ) 
            else:
                raise ValueError(f'Invalid genot noise {self.genot.noise} given.')
            
            self.x0_add_alpha = self.genot.get('x0_add_alpha', None)
        else:
            self.noise_genot = None
            self.x0_add_alpha = None 

    @staticmethod
    def compute_flow(x1: jax.Array, x0: jax.Array) -> jax.Array:
        return x1 - x0

    @staticmethod
    def compute_mu_t(x1: jax.Array, x0: jax.Array, t: float) -> jax.Array:
        return t * x1 + (1 - t) * x0

    def compute_gamma_t(self, t: float) -> jax.Array:
        if self.gamma == "bridge":
            return self.sigma * jnp.sqrt(t * (1 - t))
        elif self.gamma == "constant":
            return self.sigma
        # Added noise schedules
        elif self.gamma == "linear_decay":
            return self.sigma * (1 - t)
        elif self.gamma == "exponential_decay":
            return self.sigma * jnp.exp(-t)
        elif self.gamma == "exponential5_decay":
            return self.sigma * jnp.exp(-5*t)
        elif self.gamma == "quadratic_decay":
            return self.sigma * (1 - t**2)
        else:
            raise ValueError(f"Unknown noise schedule {self.gamma}")

    def sample_xt(
        self, x1: jax.Array, x0: jax.Array, t: float, noise: jax.Array
    ) -> jax.Array:
        mu_t = self.compute_mu_t(x1, x0, t)
        return mu_t + self.compute_gamma_t(t) * noise

    def get_batch_loss_fn(self):
        """Get single loss function."""

        def single_loss_fn(
            model: eqx.Module,
            x1: EasyDict,
            x0: EasyDict,
            t: float,
            key: jr.KeyArray,
        ) -> jax.Array:
            
            # TODO: If we change cross_attn_cond for something different than the VAE / Original image modify this
            film_cond = x0.get("embedding", None)
            cross_attn_cond=x0.data
            ###

            if self.is_genot:
                # Mask some of the conditions if we are training classifier free 
                key, subkey = jr.split(key, 2)
                classifier_free_mask = jr.uniform(key=subkey) < self.genot.classifier_free_p_uncond
                
                film_cond = jax.lax.cond(classifier_free_mask,  
                                        lambda: jnp.zeros_like(film_cond), 
                                        lambda: film_cond,
                                        )
                cross_attn_cond = jax.lax.cond(classifier_free_mask,  
                                        lambda: jnp.zeros_like(cross_attn_cond), 
                                        lambda: cross_attn_cond,
                                        )
                
                key, subkey = jr.split(key, 2)
                src_data = self.noise_genot(subkey, x0_data=x0.data, shape=x0.data.shape)        
                
                if self.x0_add_alpha is not None: # Only apply x0_add_alpha to conditional model 
                    alpha = self.x0_add_alpha 
                    src_data = jax.lax.cond(classifier_free_mask,  
                                        lambda: src_data, 
                                        lambda: alpha*x0.data + (1-alpha)*src_data,
                                        )
                         
       
                # Generating back conditioning x0 (to push towards keeping features)
                key, subkey = jr.split(key, 2)
                tgt_data = jax.lax.cond(jr.uniform(key=subkey) < self.genot.x0_prob,  
                                        lambda: x0.data, 
                                        lambda: x1.data,
                                        )
            else:
                src_data = x0.data
                tgt_data = x1.data

            noise_xt = jr.normal(key, x1.data.shape)
            
            u_t = self.compute_flow(tgt_data, src_data)
            x_t = self.sample_xt(tgt_data, src_data, t, noise_xt)


            # Notice this is only for is_genot but we need to do it later so that u_t & x_t are properly calculated
            # the self.is_genot is redundant, but we leave it for clarity
            if self.is_genot and self.x0_plus_noise:
                x_t = jnp.concatenate((x_t, x0.data), axis=0)
            
                        
            pred = model(t, x_t, 
                         film_cond=film_cond, 
                         cross_attn_cond=cross_attn_cond, 
                         key=key,
                        )

            return self.weight(t) * jnp.mean((pred - u_t) ** 2)

        def batch_loss_fn(
            model: eqx.Module,
            x1: EasyDict,
            x0: EasyDict,
            key: jr.KeyArray,
        ) -> jax.Array:
            batch_size = x1.data.shape[0]
            tkey, losskey = jr.split(key)
            losskey = jr.split(losskey, batch_size)
            # Low-discrepancy sampling over t to reduce variance
            t = jr.uniform(tkey, (batch_size,), minval=0, maxval=self.t1 / batch_size)
            t = t + (self.t1 / batch_size) * jnp.arange(batch_size)

            loss_fn = jax.vmap(ft.partial(single_loss_fn, model))
            return jnp.mean(loss_fn(x1, x0, t, losskey))

        return batch_loss_fn

    def get_train_step_fn(
        self, loss_fn: Callable, opt_update: optax.GradientTransformation
    ):
        """Returns a callable train function."""
        grad_value_loss_fn = eqx.filter_value_and_grad(loss_fn)

        @eqx.filter_jit
        def step(
            model: eqx.Module,
            x1: EasyDict,
            x0: EasyDict,
            key: jr.KeyArray,
            opt_state: optax.OptState,
        ) -> Tuple[jax.Array, eqx.Module, jr.KeyArray, optax.OptState]:
            loss, grads = grad_value_loss_fn(model, x1, x0, key)
            updates, opt_state = opt_update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            key = jr.split(key, 1)[0]
            return loss, model, key, opt_state

        return step

    def get_sample_fn(self):
        """Get single sample function."""

        @eqx.filter_jit
        def single_sample_fn(model: eqx.Module, x0: EasyDict, key=None) -> jax.Array:
            """Produce single sample from the CNF by integrating forward."""

            def func(t, x_t, args, x0=x0): 
                film_cond = x0.get("embedding", None)
                cross_attn_cond=x0.data
                
                if self.is_genot and self.x0_plus_noise:
                    x_t = jnp.concatenate((x_t, x0.data), axis=0)

                
                return model(t, x_t, 
                            film_cond=film_cond, 
                            cross_attn_cond=cross_attn_cond, 
                        )
            
            # --- 
            term = dfx.ODETerm(func)
            if self.solver == "tsit5":
                solver = dfx.Tsit5()
            elif self.solver == "euler":
                solver = dfx.Euler()
            elif self.solver == "heun":
                solver = dfx.Heun()
            else:
                raise ValueError(f"Unknown solver {self.solver}")
            if self.dt0 == 0.0:
                stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)
                dt0 = None
            else:
                stepsize_controller = dfx.ConstantStepSize()
                dt0 = self.dt0
            
            if self.is_genot:
                src_data = self.noise_genot(key, x0_data=x0.data, shape=x0.data.shape)
                if self.x0_add_alpha is not None:
                    alpha = self.x0_add_alpha 
                    src_data = alpha*x0.data + (1-alpha)*src_data
            else:
                src_data = x0.data

            sol = dfx.diffeqsolve(
                term,
                solver,
                self.t0,
                self.t1,
                dt0,
                src_data,
                stepsize_controller=stepsize_controller,
            )
            return sol.ys[0], sol.stats["num_steps"]

        return single_sample_fn
