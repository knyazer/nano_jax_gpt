import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray


class GPTConfig(eqx.Module):
    context_len: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 368
    dropout: float = 0.0


class Block(eqx.Module):
    expand_fc: eqx.nn.Linear
    proj_fc: eqx.nn.Linear
    lnorm_attn: eqx.nn.RMSNorm
    lnorm_mlp: eqx.nn.RMSNorm
    attn: eqx.nn.MultiheadAttention
    rope: eqx.nn.RotaryPositionalEmbedding

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3 = jr.split(key, 3)
        self.expand_fc = eqx.nn.Linear(config.n_embed, 4 * config.n_embed, use_bias=False, key=k1)
        self.proj_fc = eqx.nn.Linear(config.n_embed * 4, config.n_embed, use_bias=False, key=k2)
        self.lnorm_attn = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.lnorm_mlp = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.attn = eqx.nn.MultiheadAttention(config.n_head, config.n_embed, key=k3)
        self.rope = eqx.nn.RotaryPositionalEmbedding(config.n_embed, 10_000)

    def __call__(
        self, x: Float[Array, "ctx emb"], key: PRNGKeyArray | None = None
    ) -> Float[Array, "ctx emb"]:
        x = eqx.filter_vmap(self.lnorm_attn)(x)
        x = x + self.attn(
            query=self.rope(x),
            key_=self.rope(x),
            value=x,
            mask=jnp.tril(jnp.ones((x.shape[0], x.shape[0]))),
            key=key,
        )
        x = x + eqx.filter_vmap(
            lambda tok: self.lnorm_mlp(self.proj_fc(jax.nn.gelu(self.expand_fc(tok))))
        )(x)
        return x


class GPT(eqx.Module):
    config: GPTConfig = eqx.field(static=True)
    tok_embed: eqx.nn.Embedding
    blocks: list[Block]
    final_norm: eqx.nn.RMSNorm
    lm_head: eqx.nn.Linear

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.tok_embed = eqx.nn.Embedding(config.vocab_size, config.n_embed, key=k2)
        self.blocks = [Block(block_key, config) for block_key in jr.split(k3, config.n_layer)]
        self.final_norm = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.lm_head = eqx.nn.Linear(config.n_embed, config.vocab_size, use_bias=False, key=k4)
        self.config = config

    def __call__(
        self,
        idx: Int[Array, "ctx"],
        targets: Int[Array, "ctx"] | None = None,
        key: PRNGKeyArray | None = None,
    ):
        x = eqx.filter_vmap(self.tok_embed)(idx)
        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, len(self.blocks))
        for block, bkey in zip(self.blocks, keys, strict=False):  # todo: scan over layers
            x = block(x, key=bkey)
        x = eqx.filter_vmap(self.final_norm)(x)
        if targets is not None:
            logits = eqx.filter_vmap(self.lm_head)(x)
            labels = jax.nn.one_hot(targets, self.config.vocab_size)
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.sum(labels * log_probs) / (self.config.context_len - 1)
        else:
            logits = self.lm_head(x[-1])
            loss = None
        return logits, loss

    def generate(self, idx, key, max_new_tokens=30):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size <= self.config.context_len else idx[-self.config.context_len :]
            )
            logits, _ = self(idx_cond)
            idx_next = jr.categorical(logits=logits, key=key)
            idx = jnp.concat((idx, idx_next[None, ...]))

        return idx
