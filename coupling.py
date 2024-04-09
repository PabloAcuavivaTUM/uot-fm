import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Deactivate GPU JaX in local

from utils.datasets import celeba_attribute
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from typing import Tuple, Callable
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
from diffusers import FlaxAutoencoderKL
import jax.numpy as jnp


def central_crop(image: tf.Tensor, size: int) -> tf.Tensor:
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)

def process_ds(x: np.ndarray) -> tf.Tensor:
    x = tf.cast(x, tf.float32) / 127.5 - 1.0
    x = tf.image.resize(x, [313, 256], antialias=True)
    x = central_crop(x, size=256)
    x = tf.transpose(x, perm=[2, 0, 1])
    return x


def get_vae_fns(shard: jax.sharding.Sharding) -> Tuple[Callable, Callable]:
    fx_path = "CompVis/stable-diffusion-v1-4"
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        fx_path, subfolder="vae", revision="flax", dtype=jnp.float32
    )
    # replicate vae params across all devices
    vae_params = jax.device_put(vae_params, shard.replicate())

    @jax.jit
    def encode_fn(image_batch: jax.Array) -> jax.Array:
        latent_out = vae.apply({"params": vae_params}, image_batch, method=vae.encode)
        latent = latent_out.latent_dist.mode()
        latent = (latent * vae.config.scaling_factor).transpose(0, 3, 1, 2)
        return jax.lax.with_sharding_constraint(latent, shard)

    @jax.jit
    def decode_fn(latent_batch: jax.Array) -> jax.Array:
        image_out = vae.apply(
            {"params": vae_params},
            latent_batch / vae.config.scaling_factor,
            method=vae.decode,
        )
        return jax.lax.with_sharding_constraint(image_out.sample, shard)

    return encode_fn, decode_fn



if __name__ == '__main__':
    # Load normal dataset 
    N = 25_000
    celebaX, celebaY, _, _ = celeba_attribute(
        split='train',
        attribute_id=15,
        map_forward=True,
        batch_size=256,
        overfit_to_one_batch = False,
        nsamples = N,
    )

    # We need to transpose to get it into correct format for plotting (as explore internally transposes to get them all into the format)
    celebaX = celebaX.transpose(0, 3, 1, 2)
    celebaY = celebaY.transpose(0, 3, 1, 2)
    
    num_devices = len(jax.devices())
    # shard needs to have same number of dimensions as the input
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
    shard = sharding.PositionalSharding(devices)
    vae_encode_fn, vae_decode_fn = get_vae_fns(shard)

    # Load embedded dataset
    celeba_embX, celeba_embY, celeba_labelX, celeba_labelY = celeba_attribute(
        split='train',
        attribute_id=15,
        map_forward=True,
        batch_size=256,
        overfit_to_one_batch = False,
        nsamples = N,
        vae_encode_fn = vae_encode_fn,
        preprocess_fn = process_ds,
    )

    celeba_labelX[celeba_labelX == -1] = 0
    celeba_labelY[celeba_labelY == -1] = 0
    
    from utils.costs_fn_metrics import explore_cost_fn
    import ott.geometry.costs as costs
    from utils.ot_cost_fns import CoulombCost, HistCost

    metrics, comparison_metrics = explore_cost_fn(
        X=celeba_embX,
        labelX=celeba_labelX,
        Y=celeba_embY,
        labelY=celeba_labelY,
        cost_fn=[
            costs.SqEuclidean(),
            HistCost(),
            costs.PNormP(p=1),
            costs.Euclidean(),
            costs.Cosine(),
            CoulombCost(),
            costs.ElasticL1(),
            costs.ElasticL2(),
            costs.ElasticSTVS(),
        ],
        sinkhorn_matching_kwargs=dict(
            tau_a=1.0,
            tau_b=1.0,
        ),
        nbatches=50,
        batch_size=256,
        summarize=True,
        save_folder=os.path.join("compare_cost_fn", "celeba_ot_batch256"),
        overwrite=True,
        decodedX=celebaX, 
        decodedY=celebaY,
    )