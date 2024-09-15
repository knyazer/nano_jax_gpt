import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from attn import FlashMultiheadAttention
from configs import GPTConfig


class Block(eqx.Module):
    expand_fc: eqx.nn.Linear
    proj_fc: eqx.nn.Linear
    lnorm_attn: eqx.nn.LayerNorm
    lnorm_mlp: eqx.nn.LayerNorm
    attn: FlashMultiheadAttention
    dropout: eqx.nn.Dropout
    dtype: jnp.dtype

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3 = jr.split(key, 3)
        self.dtype = config.dtype
        self.expand_fc = eqx.nn.Linear(config.n_embed, 4 * config.n_embed, key=k1, dtype=self.dtype)
        self.proj_fc = eqx.nn.Linear(config.n_embed * 4, config.n_embed, key=k2, dtype=self.dtype)
        self.lnorm_attn = eqx.nn.LayerNorm(config.n_embed)
        self.lnorm_mlp = eqx.nn.LayerNorm(config.n_embed)
        self.attn = FlashMultiheadAttention(
            config.n_heads,
            config.n_embed,
            dropout_p=config.dropout,
            key=k3,
            dtype=self.dtype,
        )
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(
        self, x: Float[Array, "ctx emb"], key: PRNGKeyArray | None = None
    ) -> Float[Array, "ctx emb"]:
        if key is None:
            mlp_key, attn_key = None, None
        else:
            mlp_key, attn_key = jr.split(key)
        x_normed = eqx.filter_vmap(self.lnorm_attn)(x).astype(self.dtype)
        x = x + self.attn(
            query=x_normed,
            key_=x_normed,
            value=x_normed,
            key=attn_key,
        )

        def _mlp(x):
            x_expanded = self.expand_fc(self.lnorm_mlp(x))
            return self.proj_fc(jax.nn.gelu(x_expanded))

        x = x + self.dropout(
            eqx.filter_vmap(_mlp)(x),
            key=mlp_key,
        )
        return x


class GPT(eqx.Module):
    config: GPTConfig = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    tok_embed: eqx.nn.Embedding
    pos_embed: eqx.nn.Embedding
    blocks: list[Block]
    final_norm: eqx.nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.dtype = config.dtype
        self.tok_embed = eqx.nn.Embedding(
            config.vocab_size, config.n_embed, key=k2, dtype=self.dtype
        )
        self.pos_embed = eqx.nn.Embedding(
            config.context_len, config.n_embed, key=k1, dtype=self.dtype
        )
        self.blocks = [Block(block_key, config) for block_key in jr.split(k3, config.n_layers)]
        self.final_norm = eqx.nn.LayerNorm(config.n_embed, use_bias=False, use_weight=False)
        self.config = config

    def lm_head(self, x):
        return x @ self.tok_embed.weight.T

    def __call__(
        self,
        idx: Int[Array, "ctx"],
        targets: Int[Array, "ctx"] | None = None,
        key: PRNGKeyArray | None = None,
    ):
        x = eqx.filter_vmap(self.tok_embed)(idx) + eqx.filter_vmap(self.pos_embed)(
            jnp.arange(len(idx))
        )
        if key is None:
            key = jr.PRNGKey(0)  # inference, i guess, so dummy key
        keys = jr.split(key, len(self.blocks))
        for block, bkey in zip(self.blocks, keys):  # todo: scan over layers
            x = block(x, key=bkey)
        x = eqx.filter_vmap(self.final_norm)(x).astype(self.dtype)
        if targets is not None:
            logits = eqx.filter_vmap(self.lm_head)(x)
            labels = jax.nn.one_hot(targets, self.config.vocab_size, dtype=self.dtype)
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.sum(labels * log_probs) / self.config.context_len
        else:
            logits = self.lm_head(x[-1])
            loss = None
        return logits, loss

    def generate(self, idx, key, max_new_tokens=64):
        forward = eqx.nn.inference_mode(self)

        for _ in range(max_new_tokens):
            logits, _ = forward(idx, key=key)
            idx_next = jr.categorical(logits=logits, key=key)
            idx = jnp.concatenate([idx, jnp.array([idx_next])])
            yield idx_next
