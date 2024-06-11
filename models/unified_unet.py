# Base UNet implementation in Equinox taken from https://docs.kidger.site/equinox/examples/unet/
# Modifications to implement CrossAttention & FiLM conditioning

import math
from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

def upsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H, 1, W, 1])
    y = jnp.tile(y, [1, 1, factor, 1, factor])
    return jnp.reshape(y, [C, H * factor, W * factor])


def downsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H // factor, factor, W // factor, factor])
    return jnp.mean(y, axis=[2, 4])


def exact_zip(*args):
    _len = len(args[0])
    for arg in args:
        assert len(arg) == _len
    return zip(*args)


def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jr.split(key)

###

class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb
    
####

class LinearTimeSelfAttention(eqx.Module):
    group_norm: eqx.nn.GroupNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        dim,
        key,
        heads=4,
        dim_head=32,
    ):
        keys = jax.random.split(key, 2)
        self.group_norm = eqx.nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=keys[0])
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=keys[1])

    def __call__(self, x):
        c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "(qkv heads c) h w -> qkv heads c (h w)", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn,hen->hde", k, v)
        out = jnp.einsum("hde,hdn->hen", context, q)
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)

class Residual(eqx.Module):
    fn: Union[LinearTimeSelfAttention]

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

####

class LinearTimeCrossAttention(eqx.Module):
    """Conditional Linear Cross Attention Block."""

    group_norm: eqx.nn.GroupNorm
    heads: int
    to_kv: eqx.nn.Conv2d
    to_q: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        input_dim,
        cond_dim,
        key,
        heads=4,
        dim_head=32,
    ):
        keys = jax.random.split(key, 3)
        self.group_norm = eqx.nn.GroupNorm(min(input_dim // 4, 32), input_dim)
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_kv = eqx.nn.Conv2d(cond_dim, hidden_dim * 2, 1, key=keys[0])
        self.to_q = eqx.nn.Conv2d(input_dim, hidden_dim, 1, key=keys[1])

        self.to_out = eqx.nn.Conv2d(hidden_dim, input_dim, 1, key=keys[2])

    def __call__(self, x, cond, key=None):
        c, h, w = x.shape
        x = self.group_norm(x)
        kv = self.to_kv(cond)
        q = self.to_q(x)  # C_q, H_q, W_q

        k, v = rearrange(
            kv,
            "(kv heads c) h w -> kv heads c (h w)",
            heads=self.heads,
            kv=2,
        )

        q = rearrange(
            q,
            "(heads c) h w -> heads c (h w)",
            heads=self.heads,
        )

        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn,hen->hde", k, v)

        out = jnp.einsum("hde,hdn->hen", context, q)
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )

        return self.to_out(out)

class CondResidual(eqx.Module):
    fn: Union[LinearTimeCrossAttention]

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, cond,  *args, **kwargs):
        return self.fn(x, cond, *args, **kwargs) + x

####

class FiLM(eqx.Module):
    """Feature-wise linear modulation"""
    cond_dim: int
    channel_features: int
    fc: eqx.nn.Linear

    def __init__(self, cond_dim: int, channel_features: int, *, key):
        self.cond_dim = cond_dim
        self.channel_features = channel_features
        self.fc = eqx.nn.Linear(
            in_features=cond_dim, out_features=2 * channel_features, key=key
        )
        

    def __call__(self, x, cond):
        ch = self.fc(cond)
        gamma = ch[: self.channel_features, None, None]
        beta = ch[self.channel_features :, None, None]

        return (1 + gamma) * x + beta
    
###

class ResnetBlock(eqx.Module):
    ####
    # Main ResnetBlock
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: list[Union[Callable, eqx.nn.Linear]]
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]
    block1_groupnorm: Optional[eqx.nn.GroupNorm]
    block1_layernorm : Optional[eqx.nn.LayerNorm]
    block1_conv: eqx.nn.Conv2d
    block2_layers: list[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    ###
    # Self Attention
    attn: Optional[Residual]
    ###
    # FiLM 
    film: Optional[FiLM]
    
    ###
    # CrossAttention 
    # Self Attention
    cross_attn: Optional[Residual]

    def __init__(
        self,
        dim_in,
        dim_out,
        is_biggan,
        up,
        down,
        time_emb_dim,
        dropout_rate,
        is_attn,
        heads,
        dim_head,
        is_film,
        film_cond_dim,
        is_cross_attn,
        cross_attn_dim,
        h, 
        w, 
        *,
        key,
        
 
    ):
        
        if is_film:
            key, film_key = jax.random.split(key, 2)
            self.film = FiLM(
            cond_dim=film_cond_dim,
            channel_features=dim_in,
            key=film_key,
            )
        else:
            self.film = None 


        key, main_block_keys = jax.random.split(key, 2)
        main_block_keys = jax.random.split(main_block_keys, 7)
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [
            jax.nn.silu,
            eqx.nn.Linear(time_emb_dim, dim_out, key=main_block_keys[0]),
        ]
        
        # 2nd Test remove
        if is_film:
            self.block1_layernorm = None # eqx.nn.LayerNorm((256, h,w))
            # self.block1_groupnorm = None
        else:
            self.block1_layernorm = None 
        
        self.block1_groupnorm = eqx.nn.GroupNorm(min(max(1, dim_in // 4), 32), dim_in)

            
        self.block1_conv = eqx.nn.Conv2d(dim_in, dim_out, 3, padding=1, key=main_block_keys[1])
        
        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
            eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=main_block_keys[2]),
        ]

        assert not self.up or not self.down

        if is_biggan:
            if self.up:
                self.scaling = upsample_2d
            elif self.down:
                self.scaling = downsample_2d
            else:
                self.scaling = None
        else:
            if self.up:
                self.scaling = eqx.nn.ConvTranspose2d(
                    dim_in,
                    dim_in,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=main_block_keys[3],
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=main_block_keys[4],
                )
            else:
                self.scaling = None
        # For DDPM Yang use their own custom layer called NIN, which is
        # equivalent to a 1x1 conv
        self.res_conv = eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=main_block_keys[5])

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(
                    dim_out,
                    heads=heads,
                    dim_head=dim_head,
                    key=main_block_keys[6],
                )
            )
        else:
            self.attn = None


        if is_cross_attn:
            key, cross_attn_key = jax.random.split(key, 2)
            self.cross_attn = CondResidual(
                LinearTimeCrossAttention(
                input_dim=dim_out,
                cond_dim=cross_attn_dim, 
                heads=heads,
                dim_head=dim_head,
                key=cross_attn_key,
                ) 
            )
        else:
            self.cross_attn = None


    def __call__(self, x, t, film_cond, cross_attn_cond, *, key):
        C, _, _ = x.shape
        # In DDPM, each set of resblocks ends with an up/down sampling. In
        # biggan there is a final resblock after the up/downsampling. In this
        # code, the biggan approach is taken for both.
        # norm -> nonlinearity -> up/downsample -> conv follows Yang
        # https://github.dev/yang-song/score_sde/blob/main/models/layerspp.py

        # 1st Way to do it. Almost everything is done that way (Minimal changes)
        h = self.block1_groupnorm(x)
        if self.film is not None:
            h = self.film(h, cond=film_cond)

        # Trials different ways
        # 2nd

        # if self.film is None:
        #     h = self.block1_groupnorm(x)
        # else:
        #     h = self.block1_layernorm(x)
        #     h = self.film(h, cond=film_cond)


        h = jax.nn.silu(h)
        # 3rd Way, after SILU normalization 
        # if self.film is not None:
        #     h = self.film(h, cond=film_cond)


        if self.up or self.down:
            h = self.scaling(h)  # pyright: ignore
            x = self.scaling(x)  # pyright: ignore
        h = self.block1_conv(h)

        # 4rd Way, after block_comb normalization (Probably not that much sense as there is a ormalization just after, but it is a GroupNorm...) 
        # if self.film is not None:
        #     h = self.film(h, cond=film_cond)


        for layer in self.mlp_layers:
            t = layer(t)
        h = h + t[..., None, None]

        # # 5th 
        # if self.film is not None:
        #     h = self.film(h, cond=film_cond)

        for layer in self.block2_layers:
            # Precisely 1 dropout layer in block2_layers which requires a key.
            if isinstance(layer, eqx.nn.Dropout):
                h = layer(h, key=key)
            else:
                h = layer(h)

        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)

        out = (h + x) / jnp.sqrt(2)
        
        if self.attn is not None:
            out = self.attn(out)
        
        if self.cross_attn is not None:
            out = self.cross_attn(out, cross_attn_cond)
        
        return out



class UNet(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: eqx.nn.Conv2d
    down_res_blocks: list[list[ResnetBlock]]
    mid_block1: ResnetBlock
    mid_block2: ResnetBlock
    ups_res_blocks: list[list[ResnetBlock]]
    final_conv_layers: list[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]
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
        film_resolutions_up : list[int],
        film_resolutions_down : list[int],
        film_down: list[bool],
        film_up: list[bool],
        film_middle: list[bool],
        film_cond_dim : int, 
        cross_attn_resolutions : list[int], 
        cross_attn_dim : int,  
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

        h, w = in_height, in_width
        self.down_res_blocks = []
        num_keys = len(in_out) * num_res_blocks - 1
        keys_resblock = jr.split(keys[2], num_keys)
        
        i = 0
        for ind, (dim_in, dim_out) in enumerate(in_out):            
            is_attn = (h in attn_resolutions and w in attn_resolutions)
            is_film_resolution = (h in film_resolutions_down and w in film_resolutions_down)
            is_cross_attn = (h in cross_attn_resolutions and w in cross_attn_resolutions)
            
            res_blocks = [
                ResnetBlock(
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
                    is_film=is_film_resolution and film_down[0],
                    film_cond_dim=film_cond_dim,
                    is_cross_attn=is_cross_attn,
                    cross_attn_dim=cross_attn_dim,
                    key=keys_resblock[i],
                    h=h,
                    w=w,
                )
            ]
            i += 1
            for i_res_block in range(num_res_blocks - 2):
                res_blocks.append(
                    ResnetBlock(
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
                        is_film=is_film_resolution and film_down[i_res_block+1],
                        film_cond_dim=film_cond_dim,
                        is_cross_attn=is_cross_attn,
                        cross_attn_dim=cross_attn_dim,
                        key=keys_resblock[i],
                        h=h,
                        w=w,
                    )
                )
                i += 1

            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
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
                        is_film=is_film_resolution and film_down[num_res_blocks-1],
                        film_cond_dim=film_cond_dim,
                        is_cross_attn=is_cross_attn,
                        cross_attn_dim=cross_attn_dim,
                        key=keys_resblock[i],
                        h=h, 
                        w=w,
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(res_blocks)
        assert i == num_keys

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
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
            is_film=film_middle[0],
            film_cond_dim=film_cond_dim,
            is_cross_attn=False,
            cross_attn_dim=cross_attn_dim,
            key=keys[3],
            h=h,
            w=w,
        )
        self.mid_block2 = ResnetBlock(
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
            is_film=film_middle[1],
            film_cond_dim=film_cond_dim,
            is_cross_attn=False,
            cross_attn_dim=cross_attn_dim,
            key=keys[4],
            h=h,
            w=w, 
        )

        self.ups_res_blocks = []
        num_keys = len(in_out) * (num_res_blocks + 1) - 1
        keys_resblock = jr.split(keys[5], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_attn = (h in attn_resolutions and w in attn_resolutions)
            is_film_resolution = (h in film_resolutions_up and w in film_resolutions_up)
            is_cross_attn = (h in cross_attn_resolutions and w in cross_attn_resolutions)
            
            res_blocks = []
            for i_res_block in range(num_res_blocks - 1):
                res_blocks.append(
                    ResnetBlock(
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
                        is_film=is_film_resolution and film_up[0],
                        film_cond_dim=film_cond_dim,
                        is_cross_attn=is_cross_attn,
                        cross_attn_dim=cross_attn_dim,
                        key=keys_resblock[i],
                        h=h,
                        w=w,
                    )
                )
                i += 1
            res_blocks.append(
                ResnetBlock(
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
                    is_film=is_film_resolution and film_up[i_res_block+1],
                    film_cond_dim=film_cond_dim,
                    is_cross_attn=is_cross_attn,
                    cross_attn_dim=cross_attn_dim,
                    h=h,
                    w=w,
                )
            )
            i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
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
                        is_film=is_film_resolution and film_up[num_res_blocks],
                        film_cond_dim=film_cond_dim,
                        is_cross_attn=is_cross_attn,
                        cross_attn_dim=cross_attn_dim,
                        key=keys_resblock[i],
                        h=h,
                        w=w, 
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

    def __call__(self, t, x_t, film_cond, cross_attn_cond, *, key=None):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        h = self.first_conv(x_t)
        hs = [h]
        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                h = res_block(h, t, film_cond, cross_attn_cond, key=subkey)
                hs.append(h)

        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, t, film_cond, cross_attn_cond, key=subkey)
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, t, film_cond, cross_attn_cond, key=subkey)

        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                if res_block.up:
                    h = res_block(h, t, film_cond, cross_attn_cond, key=subkey)
                else:
                    h = res_block(jnp.concatenate((h, hs.pop()), axis=0), t, film_cond, cross_attn_cond, key=subkey)

        assert len(hs) == 0

        for layer in self.final_conv_layers:
            h = layer(h)
        return h



if __name__ == '__main__':
    import jax.random as jr
    
    test_normal_unet = False
    test_film = False
    test_cond_attention = False
    test_full_cond = True 
    
    if test_normal_unet:
        key = jr.key(seed=0)
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(4, 32, 32))
        key, subkey = jax.random.split(key)
        unet = UNet(
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
                    cross_attn_resolutions=[],
                    cross_attn_dim=0,
                    film_resolutions=[],
                    film_cond_dim=0, 
        )
        
        cross_attn_cond=None
        film_cond=None
        output = unet(t=0.5, x_t=random_input, film_cond=film_cond, cross_attn_cond=cross_attn_cond, key=subkey,)
        print(output.shape)

    if test_cond_attention:
        key = jr.key(seed=0)
        cross_attn_dim = 16
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(4, 32, 32))
        key, subkey = jax.random.split(key)
        unet = UNet(
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
                    cross_attn_resolutions=[32, 16, 8],
                    cross_attn_dim=cross_attn_dim,
                    film_resolutions=[],
                    film_cond_dim=0, 
        )
        
        cross_attn_cond=jax.random.normal(subkey, shape=(cross_attn_dim, 32, 32))
        film_cond=None
        output = unet(t=0.5, 
                      x_t=random_input, 
                      film_cond=film_cond, 
                      cross_attn_cond=cross_attn_cond, 
                      key=subkey,
                    )
        
        print(output.shape)

    if test_film:
        key = jr.key(seed=0)
        film_cond_dim = 64
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(4, 32, 32))
        key, subkey = jax.random.split(key)
        unet = UNet(
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
                    cross_attn_resolutions=[],
                    cross_attn_dim=0,
                    film_resolutions=[32, 16, 8],
                    film_cond_dim=film_cond_dim, 
        )
        
        cross_attn_cond=None
        film_cond=jax.random.normal(subkey, shape=(film_cond_dim,))
        output = unet(t=0.5, 
                      x_t=random_input, 
                      film_cond=film_cond, 
                      cross_attn_cond=cross_attn_cond, 
                      key=subkey,
                    )
        
        print(output.shape)

    if test_full_cond:
        key = jr.key(seed=0)
        cross_attn_dim = 16
        film_cond_dim = 64
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(4, 32, 32))
        key, subkey = jax.random.split(key)
        unet = UNet(
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
                    cross_attn_resolutions=[32, 16, 8],
                    cross_attn_dim=cross_attn_dim,
                    film_resolutions=[32, 16, 8],
                    film_cond_dim=film_cond_dim, 
        )
        
        cross_attn_cond=jax.random.normal(subkey, shape=(cross_attn_dim, 32, 32))
        film_cond=jax.random.normal(subkey, shape=(film_cond_dim,))
        output = unet(t=0.5, 
                      x_t=random_input, 
                      film_cond=film_cond, 
                      cross_attn_cond=cross_attn_cond, 
                      key=subkey,
                    )
        
        print(output.shape)