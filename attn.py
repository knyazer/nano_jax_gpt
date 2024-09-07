import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module, field
from equinox.nn import Dropout, Linear
from jaxtyping import Array, Float, PRNGKeyArray


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    return jnp.float32


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    dropout: Dropout | None = None,
    *,
    key: PRNGKeyArray | None = None,
    inference: bool | None = None,
) -> Float[Array, "q_seq v_size"]:
    attn = jax.nn.dot_product_attention(
        query[None, :], key_[None, :], value[None, :], is_causal=True, implementation="cudnn"
    )
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
    qk_size: int = field(static=True)
    vo_size: int = field(static=True)
    use_query_bias: bool = field(static=True)
    use_key_bias: bool = field(static=True)
    use_value_bias: bool = field(static=True)
    use_output_bias: bool = field(static=True)

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        *,
        key_size: int | None = None,
        value_size: int | None = None,
        output_size: int | None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        dtype=None,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        self.query_proj = Linear(
            query_size,
            num_heads * qk_size,
            use_bias=use_query_bias,
            dtype=dtype,
            key=qkey,
        )
        self.key_proj = Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, dtype=dtype, key=kkey
        )
        self.value_proj = Linear(
            value_size,
            num_heads * vo_size,
            use_bias=use_value_bias,
            dtype=dtype,
            key=vkey,
        )
        self.output_proj = Linear(
            num_heads * vo_size,
            output_size,
            use_bias=use_output_bias,
            dtype=dtype,
            key=okey,
        )
        self.dropout = Dropout(dropout_p, inference=inference)

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

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
