import jax
import einops
import wandb

import math 
from PIL import Image

import jax.numpy as jnp 
import numpy as np  

from typing import Any, Tuple, Union, Optional, List 
from jax.tree_util import register_pytree_node


def jnp_to_float(arr : jax.Array):
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        return arr.astype(jnp.float32)
    return arr


def jx_device_put(x : Union[jax.Array, 'EasyDict'], shard : jax.sharding.Sharding) -> jax.Array:
    if isinstance(x, EasyDict):
        return x.device_put(shard)
    

    num_devices, *_ = shard.shape
    return jax.device_put(x, shard.reshape(num_devices, *[1 for _ in x.shape[1:]]))
    
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax. If all elemenets are arrays, it also allows for slicing and jnp conversion."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, dict):
            return EasyDict(value)
        return value
    
    def slice(self, key: Union[slice, Tuple[slice]]) -> 'EasyDict':
        new_dict = EasyDict()
        for k, v in self.items():
            new_dict[k] = v[key]
        return new_dict
    
    def concatenate(self, other : 'EasyDict', axis : int =0) -> 'EasyDict':
        new_dict = EasyDict()
        for k, v in self.items():
            new_dict[k] = jnp.concatenate((v, other[k]), axis=axis)
        return new_dict

    
    def to_jnp(self, enforce_type=None) -> 'EasyDict':
        if enforce_type is None:
            return EasyDict(**{k: jnp.array(v) for k,v in self.items()})
        return EasyDict(**{k: jnp.array(v).astype(enforce_type) for k,v in self.items()})
    
    def device_put(self, shard : jax.sharding.Sharding) -> 'EasyDict':
        return jax.tree_util.tree_map(lambda x: jx_device_put(x, shard), self)
        
def unzip2(pairs, reversed : bool = False):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  if reversed:
    lst1, lst2 = lst2, lst1 

  return lst1, lst2

register_pytree_node(EasyDict,
    # Instructs JAX what are the children nodes.
    lambda d:  list(map(tuple, unzip2(sorted(d.items(), key=lambda x: x[0]), reversed=True))),     
    # Instructs JAX how to pack back into a EasyDict.
    lambda keys, vals: EasyDict(zip(keys, vals))
    )   



#########################################################################
# Logging functions for wand
def generate_wb_image(samples : jax.Array, inputs : Optional[jax.Array] = None, num_samples : int = 8) -> wandb.Image:
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
    
    return wandb.Image(np.array(image_grid))

def generate_multisample_wb_image(samples : List[jax.Array], inputs : jax.Array, num_samples : int = 16):
    images = [inputs] + samples
    stacked_images = jnp.stack(images, axis=1)[:num_samples]
    grid_image = einops.rearrange(stacked_images, 'rows cols c h w -> (rows h) (cols w) c')
    return wandb.Image(np.array(grid_image))

def to_int_color(color: Tuple[float, float, float]):
    return (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

def combine_images(images: List[Image.Image], layout: str = "horizontal", border: int = 10, background: str = "white"):
    if layout not in {"horizontal", "vertical", "grid"}:
        raise ValueError("Invalid layout. Choose from 'horizontal', 'vertical', or 'grid'.")

    # Calculate dimensions for each layout type
    if layout == "horizontal":
        total_width = sum(img.width for img in images) + border * (len(images) - 1)
        max_height = max(img.height for img in images)
        combined_image = Image.new("RGB", (total_width, max_height), background)

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + border

    elif layout == "vertical":
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images) + border * (len(images) - 1)
        combined_image = Image.new("RGB", (max_width, total_height), background)

        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height + border

    elif layout == "grid":
        # Determine grid size (square-like layout)
        grid_size = math.ceil(math.sqrt(len(images)))
        cell_width = max(img.width for img in images)
        cell_height = max(img.height for img in images)
        grid_width = grid_size * cell_width + (grid_size - 1) * border
        grid_height = grid_size * cell_height + (grid_size - 1) * border
        combined_image = Image.new("RGB", (grid_width, grid_height), background)

        for idx, img in enumerate(images):
            row, col = divmod(idx, grid_size)
            x_offset = col * (cell_width + border)
            y_offset = row * (cell_height + border)
            combined_image.paste(img, (x_offset, y_offset))

    return combined_image
