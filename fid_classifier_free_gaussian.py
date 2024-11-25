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

Ws = [-1, 0, 0.1, 0.5, 1, 2, 3, 5, 6.5, 10]
n_images_per_row = len(Ws)+1
NSAMPLES = None
batch_size = 256

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

config.name = "celeba256-male-genot-otclip-FiLM(All)-AddGaussian(0.1)-classifier-free"
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
    vae_encode_fn, vae_decode_fn = get_vae_fns(shard, config.model.get("vae_fns", "legacy"))


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
params = eqx.tree_deserialise_leaves(os.path.join(check_folder_tree, f"latest_inference_params.eqx"), params)

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
                
                zero_cond = jnp.zeros_like(x0.embedding)
                v0 = model(t, x_t, 
                            film_cond=zero_cond, 
                            cross_attn_cond=None, 
                        )
                
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
                    src_data = 0.1*x0.data + 0.9*jr.normal(key, shape=x0.data.shape)
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



# Compute metrics for complete 
eval_key = jr.PRNGKey(42)
# Notice -1 is just a cheaty way to generate from the unconditional model
Ws = [-1, 0, 0.1, 0.5, 1, 2, 3, 5, 6.5, 10]
n_images_per_row = len(Ws)+1    
eval_key, eval_key_sample = jr.split(eval_key)

metrics_all = dict()
for w in Ws:
    fs = FlowSolverClassifierFree(
            t1=config.t1,
            dt0=config.dt0,
            w=w,
            flow_sigma=config.training.flow_sigma,
            gamma=config.training.gamma,
            weight=lambda t: 1.0,
            solver=config.solver,
            is_genot=config.training.is_genot
        )
    
    sample_fn = fs.get_sample_fn()
    partial_sample_fn = partial(sample_fn, inference_model)

    eval_key, metrics_key = jr.split(eval_key)
    metric_computer = MetricComputer(
                config=config,
                shard=shard,
                eval_ds=eval_src_ds,
                sample_fn=sample_fn,
                vae_encode_fn=vae_encode_fn,
                vae_decode_fn=vae_decode_fn,
                is_genot=config.training.is_genot
            )
    _metric =  metric_computer.compute_metrics(
                        inference_model, metrics_key
                    )
    print(_metric)
    metrics_all[w] = _metric

extract_metrics = [
    'fid',
    'nfe',
    'fid_15',
    'fid_35',
    'fid_17',
]

metrics_of_interest = {k: {ek: float(v.get(ek, -999.99)) for ek in extract_metrics} for k, v in metrics_all.items()}
print(metrics_of_interest)

import json
import os 

path_dir = os.path.join('classifier_free_add_gaussian', 'fid_and_metrics')
os.makedirs(path_dir, exist_ok=True)
with open(os.path.join(path_dir, 'classifier_free_w.json'), 'w') as f:
    json.dump(metrics_of_interest, f)

