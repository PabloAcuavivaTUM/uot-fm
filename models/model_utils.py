from typing import Callable, List, Tuple
from numpy.typing import ArrayLike

import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
from diffusers import FlaxAutoencoderKL
from transformers import AutoProcessor, FlaxCLIPModel


from models.mlpmixer import Mixer2d
from models.unified_unet import UNet

def get_model(
    config: ml_collections.ConfigDict, data_shape: List[int], model_key: jr.KeyArray
):
    if config.model.type == "mlpmixer":
        return Mixer2d(
            data_shape,
            patch_size=config.model.patch_size,
            hidden_size=config.model.hidden_size,
            mix_patch_size=config.model.mix_patch_size,
            mix_hidden_size=config.model.mix_hidden_size,
            num_blocks=config.model.num_blocks,
            t1=config.t1,
            key=model_key,
        )
    elif config.model.type == "unet":
        return UNet(
            data_shape,
            is_biggan=config.model.biggan_sample,
            dim_mults=config.model.dim_mults,
            hidden_size=config.model.hidden_size,
            heads=config.model.heads,
            dim_head=config.model.dim_head,
            dropout_rate=config.model.dropout,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attention_resolution,
            ###
            cross_attn_resolutions=config.model.cross_attn_resolutions,
            cross_attn_dim=config.model.cross_attn_dim,
            film_resolutions_down=config.model.film_resolutions_down,
            film_resolutions_up=config.model.film_resolutions_up, 
            film_down=config.model.film_down,
            film_up=config.model.film_up,
            film_middle=config.model.film_middle,
            film_cond_dim=config.model.film_cond_dim, 
            ###
            key=model_key,
        )
    else:
        raise ValueError(f"Unknown model type {config.model.type}")




# from models.cond_unet import CondUNetFiLM, CondUNetCrossAttention
# from models.unet import UNet

# Keep original one for now just for reference
# def get_model(
#     config: ml_collections.ConfigDict, data_shape: List[int], model_key: jr.KeyArray
# ):
#     if config.model.type == "mlpmixer":
#         return Mixer2d(
#             data_shape,
#             patch_size=config.model.patch_size,
#             hidden_size=config.model.hidden_size,
#             mix_patch_size=config.model.mix_patch_size,
#             mix_hidden_size=config.model.mix_hidden_size,
#             num_blocks=config.model.num_blocks,
#             t1=config.t1,
#             key=model_key,
#         )
#     elif config.model.type == "unet":
#         if config.training.cond:
#             if config.training.cond_method == 'film':
#                 return CondUNetFiLM(
#                     data_shape,
#                     is_biggan=config.model.biggan_sample,
#                     dim_mults=config.model.dim_mults,
#                     hidden_size=config.model.hidden_size,
#                     heads=config.model.heads,
#                     dim_head=config.model.dim_head,
#                     dropout_rate=config.model.dropout,
#                     num_res_blocks=config.model.num_res_blocks,
#                     attn_resolutions=config.model.attention_resolution,
#                     key=model_key,
#                 )
#             elif config.training.cond_method == "attention":
#                 return CondUNetCrossAttention(
#                     data_shape,
#                     is_biggan=config.model.biggan_sample,
#                     dim_mults=config.model.dim_mults,
#                     hidden_size=config.model.hidden_size,
#                     heads=config.model.heads,
#                     dim_head=config.model.dim_head,
#                     dropout_rate=config.model.dropout,
#                     num_res_blocks=config.model.num_res_blocks,
#                     attn_resolutions=config.model.attention_resolution,
#                     key=model_key,
#                 )
#             else:
#                 raise ValueError(f"Unknown conditioning method {config.training.cond_method}.")
#         else:
#             return UNet(
#                 data_shape,
#                 is_biggan=config.model.biggan_sample,
#                 dim_mults=config.model.dim_mults,
#                 hidden_size=config.model.hidden_size,
#                 heads=config.model.heads,
#                 dim_head=config.model.dim_head,
#                 dropout_rate=config.model.dropout,
#                 num_res_blocks=config.model.num_res_blocks,
#                 attn_resolutions=config.model.attention_resolution,
#                 key=model_key,
#             )
#     else:
#         raise ValueError(f"Unknown model type {config.model.type}")


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


def get_clip_fns(batch_size: int = 1024) -> Tuple[Callable, Callable]:
    fx_path = "openai/clip-vit-base-patch32"
    model = FlaxCLIPModel.from_pretrained(fx_path)
    processor = AutoProcessor.from_pretrained(fx_path)

    def encode_img_fn(imgs: ArrayLike) -> jax.Array:
        num_batches = len(imgs) // batch_size
        img_embs = []
        for i in range(num_batches + 1):
            batch = imgs[i * batch_size : (i + 1) * batch_size]
            if len(batch) > 0:
                inputs = processor(
                    images=batch, return_tensors="np", padding=True
                ).pixel_values

                img_emb = model.get_image_features(inputs)
                img_emb /= jnp.sqrt((img_emb**2).sum(axis=1)[:, None])
                img_embs.append(img_emb)
        return jnp.concatenate(img_embs, axis=0)

    def encode_text_fn(texts: List[str]) -> jax.Array:
        num_batches = len(texts) // batch_size
        text_embs = []
        for i in range(num_batches + 1):
            batch = texts[i * batch_size : (i + 1) * batch_size]
            if len(batch) > 0:
                inputs = processor(
                    text=batch, return_tensors="np", padding=True
                ).input_ids

                text_emb = model.get_text_features(inputs)
                text_emb /= jnp.sqrt((text_emb**2).sum(axis=1)[:, None])
                text_embs.append(text_emb)
        return jnp.concatenate(text_embs, axis=0)

    return encode_img_fn, encode_text_fn
