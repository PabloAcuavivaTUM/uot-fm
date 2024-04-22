import math
import os
from typing import Callable, Optional, Union

# isort: skip_file
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Deactivate GPU JaX in local

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange


def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jax.random.split(key)


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
        print(f"X: {x.shape}")
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "(qkv heads c) h w -> qkv heads c (h w)", heads=self.heads, qkv=3
        )
        print(f"K: {k.shape} | V: {k.shape}")
        print(f"Q: {q.shape}")

        k = jax.nn.softmax(k, axis=-1)
        print(f"K-softmax: {k.shape}")
        context = jnp.einsum("hdn,hen->hde", k, v)
        print(f"Context: {context.shape}")
        out = jnp.einsum("hde,hdn->hen", context, q)
        print(f"out: {out.shape}")
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )
        print(f"out rearranged: {out.shape}")
        return self.to_out(out)


class FiLM(eqx.Module):
    """Feature-wise linear modulation"""

    conditional_dim: int
    channel_features: int
    fc: eqx.nn.Linear

    def __init__(self, conditional_dim: int, channel_features: int, *, key):
        self.conditional_dim = conditional_dim
        self.channel_features = channel_features
        self.fc = eqx.nn.Linear(
            in_features=conditional_dim, out_features=2 * channel_features, key=key
        )

    def __call__(self, x, cond):
        ch = self.fc(cond)
        gamma = ch[: self.channel_features, None, None]
        beta = ch[self.channel_features :, None, None]

        return (1 + gamma) * x + beta


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


class Residual(eqx.Module):
    fn: LinearTimeSelfAttention

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(eqx.Module):
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: list[Union[Callable, eqx.nn.Linear]]
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]
    block1_groupnorm: eqx.nn.GroupNorm
    block1_conv: eqx.nn.Conv2d
    block2_layers: list[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    attn: Optional[Residual]

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
        *,
        key,
    ):
        keys = jax.random.split(key, 7)
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [
            jax.nn.silu,
            eqx.nn.Linear(time_emb_dim, dim_out, key=keys[0]),
        ]
        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)

        self.block1_conv = eqx.nn.Conv2d(dim_in, dim_out, 3, padding=1, key=keys[1])
        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
            eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=keys[2]),
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
                    key=keys[3],
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=keys[4],
                )
            else:
                self.scaling = None
        # For DDPM Yang use their own custom layer called NIN, which is
        # equivalent to a 1x1 conv
        self.res_conv = eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=keys[5])

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(
                    dim_out,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys[6],
                )
            )
        else:
            self.attn = None

    def __call__(self, x, t, *, key):
        C, _, _ = x.shape
        # In DDPM, each set of resblocks ends with an up/down sampling. In
        # biggan there is a final resblock after the up/downsampling. In this
        # code, the biggan approach is taken for both.
        # norm -> nonlinearity -> up/downsample -> conv follows Yang
        # https://github.dev/yang-song/score_sde/blob/main/models/layerspp.py
        h = jax.nn.silu(self.block1_groupnorm(x))
        if self.up or self.down:
            h = self.scaling(h)  # pyright: ignore
            x = self.scaling(x)  # pyright: ignore
        h = self.block1_conv(h)

        for layer in self.mlp_layers:
            t = layer(t)
        h = h + t[..., None, None]
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
        return out


class SimpleResnetBlock(eqx.Module):
    dim_out: int
    up: bool
    down: bool
    dropout_rate: float
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]
    block1_groupnorm: eqx.nn.GroupNorm
    block1_conv: eqx.nn.Conv2d
    block2_layers: list[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    use_full_block2: bool

    def __init__(
        self,
        dim_in,
        dim_out,
        up,
        down,
        dropout_rate,
        use_full_block2,
        *,
        key,
    ):
        keys = jax.random.split(key, 5)
        self.dim_out = dim_out
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.use_full_block2 = False
        # ---
        # ConvBlock

        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)
        self.block1_conv = eqx.nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            padding=1,
            key=keys[0],
        )

        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
        ]

        if use_full_block2:
            self
            self.block2_layers += [
                eqx.nn.Dropout(dropout_rate),
                eqx.nn.Conv2d(
                    in_channels=dim_out,
                    out_channels=dim_out,
                    kernel_size=3,
                    padding=1,
                    key=keys[1],
                ),
            ]

        # ---
        # Scaling
        assert not self.up or not self.down

        if self.up:
            self.scaling = eqx.nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=4,
                stride=2,
                padding=1,
                key=keys[2],
            )
        elif self.down:
            self.scaling = eqx.nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=3,
                stride=2,
                padding=1,
                key=keys[3],
            )
        else:
            self.scaling = None

        # ---
        # Residual part matching
        self.res_conv = eqx.nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=1,
            key=keys[4],
        )

    def __call__(self, x, *, key=None):
        C, _, _ = x.shape

        h = jax.nn.silu(self.block1_groupnorm(x))

        if self.up or self.down:
            h = self.scaling(h)  # pyright: ignore
            x = self.scaling(x)  # pyright: ignore

        h = self.block1_conv(h)

        for layer in self.block2_layers:
            # Precisely 1 dropout layer in block2_layers which requires a key when full_size.
            if isinstance(layer, eqx.nn.Dropout):
                h = layer(h, key=key)
            else:
                h = layer(h)

        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)

        out = (h + x) / jnp.sqrt(2)

        return out


class FiLMResnetBlock(eqx.Module):
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: list[Union[Callable, eqx.nn.Linear]]
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]
    block1_groupnorm: eqx.nn.GroupNorm
    block1_conv: eqx.nn.Conv2d
    block2_layers: list[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    attn: Optional[Residual]
    conditional_film_dim: int
    film: FiLM

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
        conditional_film_dim,
        *,
        key,
    ):
        keys = jax.random.split(key, 8)
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [
            jax.nn.silu,
            eqx.nn.Linear(time_emb_dim, dim_out, key=keys[0]),
        ]
        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)

        self.conditional_film_dim = conditional_film_dim
        self.film = FiLM(
            conditional_dim=self.conditional_film_dim,
            channel_features=dim_in,
            key=keys[1],
        )

        self.block1_conv = eqx.nn.Conv2d(dim_in, dim_out, 3, padding=1, key=keys[2])
        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
            eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=keys[3]),
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
                    key=keys[4],
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=keys[5],
                )
            else:
                self.scaling = None
        # For DDPM Yang use their own custom layer called NIN, which is
        # equivalent to a 1x1 conv
        self.res_conv = eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=keys[6])

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(
                    dim_out,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys[7],
                )
            )
        else:
            self.attn = None

    def __call__(self, x, t, cond, *, key):
        C, _, _ = x.shape
        # In DDPM, each set of resblocks ends with an up/down sampling. In
        # biggan there is a final resblock after the up/downsampling. In this
        # code, the biggan approach is taken for both.
        # norm -> nonlinearity -> up/downsample -> conv follows Yang
        # https://github.dev/yang-song/score_sde/blob/main/models/layerspp.py
        h = jax.nn.silu(self.film(self.block1_groupnorm(x), cond=cond))
        if self.up or self.down:
            h = self.scaling(h)  # pyright: ignore
            x = self.scaling(x)  # pyright: ignore
        h = self.block1_conv(h)

        for layer in self.mlp_layers:
            t = layer(t)
        h = h + t[..., None, None]
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
        return out


class SimpleCNN(eqx.Module):
    res_blocks: list[SimpleResnetBlock]

    def __init__(
        self,
        dim_channels: list[int],
        dropout_rate: float,
        *,
        key,
        use_full_block2: bool = True,
    ):

        self.res_blocks = []

        for dim_in, dim_out in zip(dim_channels[:-1], dim_channels[1:]):
            key, subkey = jax.random.split(key)
            self.res_blocks.append(
                SimpleResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    up=False,
                    down=True,
                    dropout_rate=dropout_rate,
                    key=subkey,
                    use_full_block2=use_full_block2,
                )
            )

    def __call__(self, x, key=None):
        h = x
        for res_block in self.res_blocks:
            key, subkey = key_split_allowing_none(key)
            h = res_block(h, key=subkey)
        if h.shape[-1] == 1:
            h = h.ravel()
        return h


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

        # Notice that in hour case that (h w), i.e height x witdh do not coincide for q and k,v
        # as one is the conditioning (q)

        # Code from Luka:

        # k = jax.nn.softmax(k, axis=-1)
        # context = jnp.einsum("hdn,hen->hde", k, v)
        # out = jnp.einsum("hde,hdn->hen", context, q)

        # out = rearrange(
        #     out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        # )

        # return self.to_out(out)

        # Torch code from Karin. We will opt for swapping the function of context (for q)
        # and k,v (for x)
        # self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        # q = self.to_q(x)
        # # If context is undefined use x
        # context = default(context, x)
        # k = self.to_k(context)
        # v = self.to_v(context)128


if __name__ == "__main__":  # Debug area
    test_film = False
    test_crossattn = True
    # ---
    key = jax.random.key(0)

    input_channel_dim = 4
    H, W = (
        32,
        32,
    )

    if test_film:
        print("Testing FiLM")
        # Test random input
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(input_channel_dim, H, W))
        dim_channels = [input_channel_dim, 8, 16, 32, 64, 128]
        key, subkey = jax.random.split(key)
        model = SimpleCNN(dim_channels, dropout_rate=0.4, key=subkey)

        print(random_input.shape)
        key, subkey = jax.random.split(key)
        output = model(random_input, key=subkey)
        print(output.shape)

        channel_features = 10
        key, subkey = jax.random.split(key)
        random_input_film = jax.random.normal(subkey, shape=(channel_features, 20, 20))
        key, subkey = jax.random.split(key)

        film_layer = FiLM(
            conditional_dim=128,
            channel_features=channel_features,
            key=subkey,
        )
        print(random_input_film.shape)
        # Verify FiLM transformation (for varios points in same channel)
        affine_transformed = film_layer(x=random_input_film * 0, cond=output)
        print(affine_transformed.shape)
        print(affine_transformed[0, :2, :2])

        affine_transformed = film_layer(x=random_input_film * 0 + 1, cond=output)
        print(affine_transformed.shape)
        print(affine_transformed[0, :2, :2])
    if test_crossattn:
        print("Testing CrossAttn")
        # Test random input
        key, subkey = jax.random.split(key)
        random_cond = jax.random.normal(subkey, shape=(input_channel_dim, H, W))

        dim_channels = [input_channel_dim, 32, 64, 128]

        key, subkey = jax.random.split(key)
        model = SimpleCNN(dim_channels, dropout_rate=0.4, key=subkey)
        print(random_cond.shape)

        key, subkey = jax.random.split(key)
        cond = model(random_cond, key=subkey)
        print(cond.shape)

        # Test random input
        input_dim = 32
        key, subkey = jax.random.split(key)
        random_input = jax.random.normal(subkey, shape=(input_dim, H, W))

        print("CrossAttentionLayer")
        key, subkey = jax.random.split(key)
        cross_attn = LinearTimeCrossAttention(
            input_dim=input_dim,
            cond_dim=dim_channels[-1],
            key=subkey,
            heads=4,
            dim_head=32,
        )
        key, subkey = jax.random.split(key)
        output = cross_attn(random_input, cond, key=subkey)
        print(
            f"Random Input: {random_input.shape} | CrossAttention Output: {output.shape}"
        )
