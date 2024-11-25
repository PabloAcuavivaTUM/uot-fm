import logging
import os

from copy import deepcopy
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.random as jr
import jax.sharding as sharding
import ml_collections
import numpy as np
import orbax.checkpoint as obx
import wandb
import os
import jax.numpy as jnp
import einops

from functools import partial, reduce
from typing import Union

from models import get_model, get_vae_fns
from utils import MetricComputer, get_loss_builder, get_translation_datasets

import matplotlib.pyplot as plt

Ws = [0, 6.5] # 
Ws = [0, 0.1, 0.5, 1, 2, 3, 5, 6.5, 10]
method = 'average_vector'
method = 'average_embedding'
n_negativess = [2,4,6,8]
def compute_mean(arr):
    return jnp.mean(arr, axis=0)

def compute_centroid(arr):
    s = jnp.sum(arr, axis=0)
    centroid = s / jnp.linalg.norm(s)
    return centroid

n_images_per_row = len(Ws)+2
NSAMPLES = 1024 # *12
batch_size = 32
batch_size_matching = 1024


def concatenate(self, other : 'EasyDict', axis : int =0) -> 'EasyDict':
    new_dict = EasyDict()
    for k, v in self.items():
        new_dict[k] = jnp.concatenate((v, other[k]), axis=axis)
    return new_dict
##################


workdir = 'runs'

from configs.base_uotfm import get_uotfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.male.base_male import get_male_config

config = get_uotfm_config()
config = get_unet_config(config)
config = get_celeba_config(config)
config = get_male_config(config)

config.training.num_steps = 600_000

config.training.tau_a = 0.95
config.training.tau_b = 0.95

config.name = "celeba256-male-genot-otclip-full-cond"
config.wandb_group = "genot"
config.training.is_genot = True

config.data.additional_embedding = "clip"
config.model.film_cond_dim = 512

config.model.film_resolutions_down = [i for i in range(200)] # This could be 4, 8, 16, 32 
config.model.film_resolutions_up = [i for i in range(200)]   # This could be 4, 8, 16, 32
config.model.film_down = [True, True, True, True] 
config.model.film_up = [True, True, True, True, True]
config.model.film_middle = [True, True]

config.training.compare_on = "embedding"
config.training.ot_cost_fn = "cosine"
######################
# Load model and VAEs
jax.config.update("jax_threefry_partitionable", True)
# create rng keys
key = jr.PRNGKey(config.seed)
np.random.seed(config.seed)
model_key, eval_key = jr.split(key, 2)
# set up sharding
num_devices = len(jax.devices())
# shard needs to have same number of dimensions as the input
devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
shard = sharding.PositionalSharding(devices)
if config.model.use_vae:
    logging.info("Loading VAE...")
    # load vae and jitted encode/decode functions
    vae_encode_fn, vae_decode_fn = get_vae_fns(shard, config.model.use_vae)


# build model and optimization functions
model = get_model(config, config.model.input_shape, model_key)
loss_builder = get_loss_builder(config)
sample_fn = loss_builder.get_sample_fn()
# create checkpoint manager
mngr_options = obx.CheckpointManagerOptions(
    create=True, max_to_keep=3, best_fn=lambda metric: metric, best_mode="min"
)
ckpt_mngr = obx.CheckpointManager(
    directory=f"{os.getcwd()}/{workdir}/{config.name}/checkpoints",
    checkpointers=obx.Checkpointer(obx.PyTreeCheckpointHandler()),
    options=mngr_options,
)
# load saved checkpoint
if config.eval.checkpoint_step is not None:
    latest_step = config.eval.checkpoint_step
else:
    latest_step = ckpt_mngr.best_step()
print(f"Loading model from step {latest_step}...")

params, static = eqx.partition(model, eqx.is_array)


step = 450_000
check_folder_tree = f"{os.getcwd()}/{workdir}/{config.name}/tree_checkpoints"
params = eqx.tree_deserialise_leaves(os.path.join(check_folder_tree, f"params{step}.eqx"), params)

# ! Restore checkpoint seems to have a problem, see how this can be solved for now use the code before from tree_deserialize
# restored_ckpt = ckpt_mngr.restore(75_000, model)
# restored_params = eqx.filter(restored_ckpt, eqx.is_array)

model = eqx.combine(params, static)
inference_model = eqx.tree_inference(model, value=True)

######
# Load dataset
config.data.nsamples = NSAMPLES
#config.data.batch_size = 256
if config.task == "translation":
    _, _, eval_src_ds, eval_tgt_ds = get_translation_datasets(
        config, shard, vae_encode_fn if config.model.use_vae else None
    )
    logging.info(f"num_eval_src: {eval_src_ds.length}")
    logging.info(f"num_eval_tgt: {eval_tgt_ds.length}")
elif config.task == "generation":
    eval_src_ds, eval_tgt_ds = None, None
####################
# Inference
import functools as ft
from typing import Any, Callable, Dict, List, Optional, Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.random as jr
from utils import EasyDict

class FlowSolverClassifierFree:
    """Class for Flow Matching loss computation and sampling."""

    def __init__(
        self,
        t1: float,
        dt0: float,
        w : float, 
        mean_fn : Callable,
        t0: float = 0.0,
        gamma: str = "constant",
        flow_sigma: Optional[float] = 0.1,
        weight: Optional[Callable[[float], float]] = lambda t: 1.0,
        solver: str = "tsit5",
        is_genot : bool = False 
        
    ):
        self.t1 = t1
        self.t0 = t0
        self.dt0 = dt0
        self.w = w
        self.gamma = gamma
        self.sigma = flow_sigma
        self.weight = weight
        self.solver = solver
        self.is_genot = is_genot
        self.mean_fn = mean_fn 

    def get_sample_fn(self):
        """Get single sample function."""

        @eqx.filter_jit
        def single_sample_fn(model: eqx.Module, x0: EasyDict, key=None, sample_array: jax.Array=None) -> jax.Array:
            """Produce single sample from the CNF by integrating forward."""

            def func(t, x_t, args, x0=x0): 
                # WARNING: Right now it only works for a model trained for FiLM embedding with no cross attention        
                u0 = model(t, x_t, 
                            film_cond=x0.get("embedding", None), 
                            cross_attn_cond=None, 
                        )
                
                if method == "average_embedding":
                    v0 = model(t, x_t, 
                                film_cond=self.mean_fn(x0.negative_embedding), 
                                cross_attn_cond=None, 
                            )
                elif method == "average_vector":
                    v0s = jax.vmap(lambda neg_emb: model(t, x_t, film_cond=neg_emb, cross_attn_cond=None))(x0.negative_embedding)
                    v0 = v0s.mean(axis=0)
                    
                else:
                    raise ValueError(f"Invalid method provided {method}.") 
                return (1+self.w)*u0 - self.w*v0
            
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
                if sample_array is None:
                    src_data = jr.normal(key, shape=x0.data.shape)
                else:
                    src_data = sample_array
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
    

def build_img(vae_img):
    if len(vae_img.shape) == 3:
        decoded = vae_decode_fn(np.expand_dims(vae_img, axis=0))
    else:
        decoded = vae_decode_fn(vae_img)
    return np.clip(decoded, -1.0, 1.0) * 0.5 + 0.5


def generate_image(samples : jax.Array, inputs : Optional[jax.Array] = None, num_samples : int = 8) -> wandb.Image:
    if inputs is None: 
        image_grid = jnp.concatenate([samples[: num_samples**2]])
        image_grid = einops.rearrange(
            image_grid,
            "(n m) c h w -> (n h) (m w) c",
            n=num_samples,
            m=num_samples,
        )
    else:
        nmb_double_rows = num_samples // 2
        rows = []
        # create image grid of alternating rows of input and output
        for row_idx in range(nmb_double_rows):
            rows.append(inputs[row_idx * num_samples : (row_idx + 1) * num_samples])
            rows.append(samples[row_idx * num_samples : (row_idx + 1) * num_samples])
        image_grid = jnp.concatenate(rows)
        image_grid = einops.rearrange(
            image_grid,
            "(n m) c h w -> (n h) (m w) c",
            n=nmb_double_rows * 2,
            m=num_samples,
        )
        
    return image_grid


#########################
# Notice this is hardcoded as a global variable!
x00 = reduce(concatenate, iter(eval_src_ds))

for n_negatives in n_negativess:

    n_images_per_row = len(Ws)+1 + n_negatives

    def plot_classifier_free_from_matching(explore_name : str, 
                        src : EasyDict, 
                        neg : EasyDict, 
                        key,
                        mean_fn : Callable, 
                        same_noise : bool = False,
                        expand_each : Optional[int] = None
                        ):
        
        # Prevent making changes to given src and neg
        src = deepcopy(src)
        neg = deepcopy(neg)
        if expand_each is not None:
            # Expand first entries until batch_size is reached
            batch_data_size = len(src.data)

            _slice = jnp.array([[k]*expand_each for k in range(batch_data_size // expand_each + 1)]).flatten()[:batch_data_size]
            src = src.slice(_slice)
            neg = neg.slice(_slice)
            

        # Set embedding to generation (FiLM conditioning only, no need to access VAE)
        src['negative_embedding'] = neg['embedding']

        x0_image = build_img(src.data)
        x0_image_negatives = jax.vmap(build_img, in_axes=1, out_axes=1)(neg.data)
        x0_image_negatives = jnp.swapaxes(x0_image_negatives, 0, 1)

        # Make so all of them are generated from the same noise
        key, noise_key = jr.split(key)
        if same_noise:
            sample_array_single = jr.normal(noise_key, shape=src.data.shape[1:])
            sample_array = jnp.stack([sample_array_single] * src.data.shape[0])
        else:
            sample_array = jr.normal(noise_key, shape=src.data.shape)

        ####
        # Guided samples
        x0_sols = []
        for w in Ws:
            fs = FlowSolverClassifierFree(
                        t1=config.t1,
                        dt0=config.dt0,
                        w=w,
                        mean_fn=mean_fn,
                        flow_sigma=config.training.flow_sigma,
                        gamma=config.training.gamma,
                        weight=lambda t: 1.0,
                        solver=config.solver,
                        is_genot=config.training.is_genot
                    )
            sample_fn = fs.get_sample_fn()

            partial_sample_fn = partial(sample_fn, inference_model)
            sample_x0, nfe = jax.vmap(partial_sample_fn)(src, key=None, sample_array=sample_array)

            # Decode it into images
            x0_sol = build_img(sample_x0)
            x0_sols.append(x0_sol)
            
        print('=============================================')
        print(len(x0_sols), x0_sols[0].shape)
        print(x0_image_negatives.shape, len(list(x0_image_negatives)), list(x0_image_negatives)[0].shape)
        print(x0_image.shape)
        print('=============================================')
        reordered_image = einops.rearrange(
                jnp.stack([x0_image] + list(x0_image_negatives) + x0_sols, axis=1),
                "n b c h w -> (n b) c h w",
            )

        for i in range(len(src.data) // n_images_per_row):
            gradient_image = generate_image(reordered_image[n_images_per_row*n_images_per_row*i:], 
                                    inputs=None, 
                                    num_samples=n_images_per_row,
                                    )

            plt.figure(figsize=(20, 20))
            plt.imshow(gradient_image)
            plt.axis('off')
            os.makedirs(os.path.join(f'classifier_free_matching_multiple_small', f"{explore_name}_{method}"), exist_ok=True)
            plt.savefig(os.path.join(f'classifier_free_matching_multiple_small', f"{explore_name}_{method}", f'sample_{i}.png'))
            plt.close()

    # Batch sampler configuration
    from ott.geometry.pointcloud import geometry, PointCloud
    from ott.solvers.linear import sinkhorn
    from utils.ot_cost_fns import  cost_fns
    ### Add cosine similarity
    import ott.geometry.costs as costs
    @jax.tree_util.register_pytree_node_class
    class CosineSimilarity(costs.CostFn):
        def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
            x_norm = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
            y_norm = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
            return jnp.sum(x_norm * y_norm, axis=-1)

    cost_fns["cosine_similarity"] = CosineSimilarity()

    ####

    matches = [('cosine_similarity', 'embedding', False, n_images_per_row), ('cosine', 'embedding', False, n_images_per_row), ('euclidean', 'data', False, n_images_per_row), ('coulomb', 'data', False, n_images_per_row),
                ('cosine_similarity', 'embedding', True, None), ('cosine', 'embedding', True, None), ('euclidean', 'data', True, None), ('coulomb', 'data', True, None),
            ('cosine_similarity', 'embedding', False, None), ('cosine', 'embedding',False, None), ('euclidean', 'data', False, None), ('coulomb', 'data', False, None),
            ]

    # For quick trials 
    # matches = [('cosine_similarity', 'embedding', False, n_images_per_row),  
    #            ('coulomb', 'data', False, n_images_per_row),
    #         ]

    all = []
    for ot_cost_fn, compare_on, same_noise, expand_each in matches:
        if compare_on == 'data':
            mean_fn = compute_mean 
        elif compare_on == 'embedding':
            mean_fn = compute_centroid
        else:
            raise ValueError('Invalid compare on')
         
        print(ot_cost_fn, compare_on, same_noise, expand_each)
        all.append((ot_cost_fn, compare_on, same_noise, expand_each))
        tau_a = 0.95
        tau_b = 0.95
        epsilon = 0.01

        def get_cost_matrix(source_batch : EasyDict, target_batch : EasyDict):
            geom = PointCloud(
                jnp.reshape(source_batch[compare_on], [batch_size_matching, -1]),
                jnp.reshape(target_batch[compare_on], [batch_size_matching, -1]),
                epsilon=epsilon,
                scale_cost=1.0,
                cost_fn=cost_fns[ot_cost_fn],
                batch_size=None,
            )
            cm = geom.cost_matrix

            if ot_cost_fn != 'coulomb':
                # Set so that it doesn't choose same sample
                cm = cm.at[jnp.arange(len(cm)), jnp.arange(len(cm))].set(2*jnp.max(cm))
            else:
                # Notice this is needed for coulomb cost to stop infinity in the diagonal
                cm = cm.at[jnp.arange(len(cm)), jnp.arange(len(cm))].set(10e4)
            return cm 



        def resample_from_matrix(source_batch : EasyDict, target_batch : EasyDict, cm, key):
            # Notice here we don't solve the OT problem but directly sample from source and take the furtherst away points
            # Only consider the first batch_size rows from cost matrix (all columns though!)
            src_indices = jnp.arange(batch_size)
            top_indices = jnp.argsort(cm, axis=1)[:batch_size, :n_negatives]
                                                    
                                                    # Shape [batch_size, n_negatives, *]
            return source_batch.slice(src_indices), target_batch.slice(top_indices) 

                        



        resample_key = jr.PRNGKey(24)
        resampled_batch_src, resampled_batch_tgt = resample_from_matrix(source_batch=x00, 
                                                                        target_batch=x00,
                                                                        cm=get_cost_matrix(source_batch=x00, target_batch=x00),
                                                                        key=resample_key,
                                                                    )

        print(all)
        
        eval_key = jr.PRNGKey(42)
        plot_classifier_free_from_matching(f'ot_{ot_cost_fn}_{compare_on}{"_same_noise" if same_noise else ""}{"_multisample" if expand_each is not None else ""}_{n_negatives}', 
                            src=resampled_batch_src, 
                            neg=resampled_batch_tgt, 
                            mean_fn=mean_fn,
                            key=eval_key,
                            same_noise = same_noise,
                            expand_each=expand_each,
                            )
        
