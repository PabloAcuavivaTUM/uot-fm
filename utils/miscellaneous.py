import jax
import einops
import wandb

import jax.numpy as jnp 
import numpy as np  

from typing import Any, Tuple, Union, Optional
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