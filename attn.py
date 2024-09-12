import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module, field, filter_jit
from equinox.nn import Dropout, Linear
from jax.experimental.pallas.ops.gpu import attention as gpu_attention
from jax.nn import dot_product_attention as fallback_dot_product_attention
from jaxtyping import Array, Float, PRNGKeyArray


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    return jnp.float32


@filter_jit
def causal_dot_product_attention(q, k, v):
    try:
        if jax.device_count(backend="gpu") > 0:
            return gpu_attention.mha(q, k, v, None, causal=True, block_q=32, block_k=32)
    except Exception:
        ...

    return fallback_dot_product_attention(q, k, v, is_causal=True)


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    dropout: Dropout | None = None,
    *,
    key: PRNGKeyArray | None = None,
    inference: bool | None = None,
) -> Float[Array, "q_seq v_size"]:
    attn = causal_dot_product_attention(query[None, :], key_[None, :], value[None, :])
    if dropout is not None:
        attn = dropout(attn, key=key, inference=inference)
    return attn


class FlashMultiheadAttention(Module, strict=True):
    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    dropout: Dropout

    num_heads: int = field(static=True)
    query_size: int = field(static=True)
    key_size: int = field(static=True)
    value_size: int = field(static=True)
    output_size: int = field(static=True)
    rope: eqx.nn.RotaryPositionalEmbedding

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        *,
        key_size: int | None = None,
        value_size: int | None = None,
        output_size: int | None = None,
        dropout_p: float = 0.0,
        inference: bool = False,
        dtype=None,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        self.rope = eqx.nn.RotaryPositionalEmbedding(query_size // num_heads, 10_000)
        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if output_size is None:
            output_size = query_size

        self.query_proj = Linear(
            query_size,
            query_size,
            use_bias=False,
            dtype=dtype,
            key=qkey,
        )
        self.key_proj = Linear(key_size, key_size, use_bias=False, dtype=dtype, key=kkey)
        self.value_proj = Linear(
            value_size,
            value_size,
            use_bias=False,
            dtype=dtype,
            key=vkey,
        )
        self.output_proj = Linear(
            query_size,
            output_size,
            use_bias=False,
            dtype=dtype,
            key=okey,
        )
        self.dropout = Dropout(dropout_p, inference=inference)

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size

    @jax.named_scope("eqx.nn.MultiheadAttention")
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "q_seq o_size"]:
        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        shape_before = query_heads.shape

        query_heads = jax.vmap(lambda x: self.rope(x).astype(x.dtype), in_axes=(1,))(query_heads)
        key_heads = jax.vmap(lambda x: self.rope(x).astype(x.dtype), in_axes=(1,))(key_heads)

        query_heads = einops.rearrange(query_heads, "num_heads seq embed -> seq num_heads embed")
        key_heads = einops.rearrange(key_heads, "num_heads seq embed -> seq num_heads embed")

        assert query_heads.shape == shape_before

        attn_key = key if key is not None else None
        attn = dot_product_attention(
            query_heads, key_heads, value_heads, key=attn_key, dropout=self.dropout
        )
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)
