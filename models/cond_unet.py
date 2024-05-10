# UNet implementation in Equinox taken from https://docs.kidger.site/equinox/examples/unet/

import math
from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
# models.cond_utils
from models.cond_utils import (
    FiLMResnetBlock,
    CrossAttentionResnetBlock,
    SimpleCNN,
    SinusoidalPosEmb,
    exact_zip,
    key_split_allowing_none,
)


class CondUNetFiLM(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv2d
    down_res_blocks: list[list[FiLMResnetBlock]]
    mid_block1: FiLMResnetBlock
    mid_block2: FiLMResnetBlock
    ups_res_blocks: list[list[FiLMResnetBlock]]
    final_conv_layers: list[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]
    
    cond_cnn : SimpleCNN
    
    def __init__(
        self,
        data_shape: tuple[int, int, int],
        is_biggan: bool,
        dim_mults: list[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: list[int],
        *,
        key,
    ):
        keys = jax.random.split(key, 8)
        del key

        data_channels, in_height, in_width = data_shape

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(exact_zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = eqx.nn.MLP(
            hidden_size,
            hidden_size,
            4 * hidden_size,
            1,
            activation=jax.nn.silu,
            key=keys[0],
        )
        self.first_conv = eqx.nn.Conv2d(
            data_channels, hidden_size, kernel_size=3, padding=1, key=keys[1]
        )
        
        
        # ! Move to arguments in init. For now testing
        cond_cnn_dim_channels  = [8, 16, 32, 64, 128]
        conditional_film_dim = cond_cnn_dim_channels[-1]
        use_full_block2 = True
        
        cond_cnn_dim_channels = [data_channels] + cond_cnn_dim_channels
        
        self.cond_cnn = SimpleCNN(dim_channels=cond_cnn_dim_channels, dropout_rate=dropout_rate, key=keys[2], use_full_block2=use_full_block2)
        
        # ---

        h, w = in_height, in_width
        self.down_res_blocks = []
        num_keys = len(in_out) * num_res_blocks - 1
        keys_resblock = jr.split(keys[3], num_keys)
        i = 0
        
        # ---
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = [
                FiLMResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                    conditional_film_dim=conditional_film_dim, 
                )
            ]
            i += 1
            for _ in range(num_res_blocks - 2):
                res_blocks.append(
                    FiLMResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        conditional_film_dim=conditional_film_dim, 
                    )
                )
                i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    FiLMResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=True,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        conditional_film_dim=conditional_film_dim,
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(res_blocks)
        assert i == num_keys

        mid_dim = dims[-1]
        self.mid_block1 = FiLMResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=True,
            heads=heads,
            dim_head=dim_head,
            key=keys[4],
            conditional_film_dim=conditional_film_dim,
        )
        self.mid_block2 = FiLMResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=False,
            heads=heads,
            dim_head=dim_head,
            key=keys[5],
            conditional_film_dim=conditional_film_dim,
        )

        self.ups_res_blocks = []
        num_keys = len(in_out) * (num_res_blocks + 1) - 1
        keys_resblock = jr.split(keys[6], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = []
            for _ in range(num_res_blocks - 1):
                res_blocks.append(
                    FiLMResnetBlock(
                        dim_in=dim_out * 2,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        conditional_film_dim=conditional_film_dim,
                    )
                )
                i += 1
            res_blocks.append(
                FiLMResnetBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_in,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                    conditional_film_dim=conditional_film_dim,
                )
            )
            i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    FiLMResnetBlock(
                        dim_in=dim_in,
                        dim_out=dim_in,
                        is_biggan=is_biggan,
                        up=True,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        conditional_film_dim=conditional_film_dim,
                    )
                )
                i += 1
                h, w = h * 2, w * 2

            self.ups_res_blocks.append(res_blocks)
        assert i == num_keys

        self.final_conv_layers = [
            eqx.nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            jax.nn.silu,
            eqx.nn.Conv2d(hidden_size, data_channels, 1, key=keys[7]),
        ]

    def __call__(self, t, x_t, x0, *, key=None):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        h = self.first_conv(x_t)
        hs = [h]
        key, subkey = key_split_allowing_none(key)
        cond = self.cond_cnn(x0, key=subkey)
        
        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                h = res_block(h, t, cond=cond, key=subkey)
                hs.append(h)
                
        

        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, t, cond=cond, key=subkey,)
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, t, cond=cond, key=subkey,)
        

        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                if res_block.up:
                    h = res_block(h, t, cond=cond, key=subkey)
                else:
                    h = res_block(jnp.concatenate((h, hs.pop()), axis=0), t, cond=cond, key=subkey)
               

        assert len(hs) == 0

        # ? Conditioning for final layers? For now off 
        for layer in self.final_conv_layers:
            h = layer(h)
        return h

class CondUNetCrossAttention(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv2d
    down_res_blocks: list[list[CrossAttentionResnetBlock]]
    mid_block1: CrossAttentionResnetBlock
    mid_block2: CrossAttentionResnetBlock
    ups_res_blocks: list[list[CrossAttentionResnetBlock]]
    final_conv_layers: list[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]
    # cond_cnn : Optional[SimpleCNN]

    def __init__(
        self,
        data_shape: tuple[int, int, int],
        is_biggan: bool,
        dim_mults: list[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: list[int],
        *,
        key,
    ):
        keys = jax.random.split(key, 7)
        del key

        data_channels, in_height, in_width = data_shape

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(exact_zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = eqx.nn.MLP(
            hidden_size,
            hidden_size,
            4 * hidden_size,
            1,
            activation=jax.nn.silu,
            key=keys[0],
        )
        self.first_conv = eqx.nn.Conv2d(
            data_channels, hidden_size, kernel_size=3, padding=1, key=keys[1]
        )
        
        # ! Move to arguments in init. For now testing
        cond_cnn_dim_channels  = [32, 64, 128]
        use_full_block2 = True
        cond_cnn_dim_channels = [data_channels] + cond_cnn_dim_channels
        cond_dim = cond_cnn_dim_channels[-1]
        
        if False: # Deactivate for now for quick testing
            self.cond_cnn = SimpleCNN(dim_channels=cond_cnn_dim_channels, dropout_rate=dropout_rate, key=keys[2], use_full_block2=use_full_block2)
        
        # ----------

        h, w = in_height, in_width
        self.down_res_blocks = []
        num_keys = len(in_out) * num_res_blocks - 1
        keys_resblock = jr.split(keys[2], num_keys)
        i = 0
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = [
                CrossAttentionResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    cond_dim=cond_dim,
                    key=keys_resblock[i],
                )
            ]
            i += 1
            for _ in range(num_res_blocks - 2):
                res_blocks.append(
                    CrossAttentionResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        cond_dim=cond_dim,
                        key=keys_resblock[i],
                    )
                )
                i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    CrossAttentionResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=True,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        cond_dim=cond_dim,
                        key=keys_resblock[i],
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(res_blocks)
        assert i == num_keys

        mid_dim = dims[-1]

        self.mid_block1 = CrossAttentionResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=True,
            heads=heads,
            dim_head=dim_head,
            cond_dim=cond_dim,
            key=keys[3],
        )
        self.mid_block2 = CrossAttentionResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=False,
            heads=heads,
            dim_head=dim_head,
            cond_dim=cond_dim,
            key=keys[4],
        )


        self.ups_res_blocks = []
        num_keys = len(in_out) * (num_res_blocks + 1) - 1
        keys_resblock = jr.split(keys[5], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = []
            for _ in range(num_res_blocks - 1):
                res_blocks.append(
                    CrossAttentionResnetBlock(
                        dim_in=dim_out * 2,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        cond_dim=cond_dim,
                        key=keys_resblock[i],
                    )
                )
                i += 1
            res_blocks.append(
                CrossAttentionResnetBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_in,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    cond_dim=cond_dim,
                    key=keys_resblock[i],
                )
            )
            i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    CrossAttentionResnetBlock(
                        dim_in=dim_in,
                        dim_out=dim_in,
                        is_biggan=is_biggan,
                        up=True,
                        down=False,
                        time_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        cond_dim=cond_dim,
                        key=keys_resblock[i],
                    )
                )
                i += 1
                h, w = h * 2, w * 2

            self.ups_res_blocks.append(res_blocks)
        assert i == num_keys

        self.final_conv_layers = [
            eqx.nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            jax.nn.silu,
            eqx.nn.Conv2d(hidden_size, data_channels, 1, key=keys[6]),
        ]
        
    def __call__(self, t, x_t, x0, *, key=None):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        h = self.first_conv(x_t)
        hs = [h]
        key, subkey = key_split_allowing_none(key)
        #cond = self.cond_cnn(x0, key=subkey)
        cond = x0 
                
        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                h = res_block(h, t, cond, key=subkey)
                hs.append(h)

        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, t, cond, key=subkey)
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, t, cond, key=subkey)

        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                if res_block.up:
                    h = res_block(h, t, cond, key=subkey)
                else:
                    h = res_block(jnp.concatenate((h, hs.pop()), axis=0), t, cond, key=subkey)

        assert len(hs) == 0

        for layer in self.final_conv_layers:
            h = layer(h)
        return h

if __name__ == '__main__':
    import jax.random as jr
    
    test_film = False
    test_cond_attention = True    
    if test_film:
        key = jr.key(seed=0)
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(4, 32, 32))
        key, subkey = jax.random.split(key)
        unet = CondUNetFiLM(
                    (4,32,32),
                    is_biggan=False,
                    dim_mults=[2,2,2],
                    hidden_size=128,
                    heads=1,
                    dim_head=64,
                    dropout_rate=0.1,
                    num_res_blocks=4,
                    attn_resolutions=[16],
                    key=subkey,
        )
        
        output = unet(t=0.5, x_t=random_input, x0=random_input, key=subkey,)
        print(output.shape)
    if test_cond_attention:
        key = jr.key(seed=0)
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(4, 32, 32))
        key, subkey = jax.random.split(key)
        unet = CondUNetCrossAttention(
                    (4,32,32),
                    is_biggan=False,
                    dim_mults=[2,2,2],
                    hidden_size=128,
                    heads=1,
                    dim_head=64,
                    dropout_rate=0.1,
                    num_res_blocks=4,
                    attn_resolutions=[16],
                    key=subkey,
        )
        
        output = unet(t=0.5, x_t=random_input, x0=random_input, key=subkey,)
        print(output.shape)
    
