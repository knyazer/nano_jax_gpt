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

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3 = jr.split(key, 3)
        self.expand_fc = eqx.nn.Linear(
            config.n_embed, 4 * config.n_embed, key=k1, dtype=config.dtype
        )
        self.proj_fc = eqx.nn.Linear(config.n_embed * 4, config.n_embed, key=k2, dtype=config.dtype)
        self.lnorm_attn = eqx.nn.LayerNorm(config.n_embed)
        self.lnorm_mlp = eqx.nn.LayerNorm(config.n_embed)
        self.attn = FlashMultiheadAttention(
            config.n_heads,
            config.n_embed,
            dropout_p=config.dropout,
            key=k3,
            dtype=config.dtype,
        )
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(
        self, x: Float[Array, "ctx emb"], key: PRNGKeyArray | None = None
    ) -> Float[Array, "ctx emb"]:
        if key is None:
            mlp_key, attn_key = None, None
        else:
            mlp_key, attn_key = jr.split(key)
        x_normed = eqx.filter_vmap(self.lnorm_attn)(x).astype(x.dtype)
        x_normed = jnp.nan_to_num(x_normed)  # make sure the softmax is well defined
        x = x + self.attn(
            query=x_normed,
            key_=x_normed,
            value=x_normed,
            key=attn_key,
        )

        def _mlp(x):
            x_expanded = self.expand_fc(self.lnorm_mlp(x).astype(x.dtype))
            return self.proj_fc(jax.nn.gelu(x_expanded))

        x = x + self.dropout(
            eqx.filter_vmap(_mlp)(x),
            key=mlp_key,
        )
        return x


class GPT(eqx.Module):
    config: GPTConfig = eqx.field(static=True)
    tok_embed: eqx.nn.Embedding
    pos_embed: eqx.nn.Embedding
    blocks: list[Block]
    final_norm: eqx.nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.tok_embed = eqx.nn.Embedding(
            config.vocab_size, config.n_embed, key=k2, dtype=config.dtype
        )
        self.pos_embed = eqx.nn.Embedding(
            config.context_len, config.n_embed, key=k1, dtype=config.dtype
        )
        self.blocks = [Block(block_key, config) for block_key in jr.split(k3, config.n_layers)]
        self.final_norm = eqx.nn.LayerNorm(config.n_embed)
        self.config = config

    @classmethod
    def make(cls, key: PRNGKeyArray, config: GPTConfig):
        model = cls(key, config)

        # now, init weights according to the gpt2 paper, with a small normal
        def _init(path, x):
            path_str = "/".join([getattr(p, "name", "") + str(getattr(p, "idx", "")) for p in path])
            key = jr.key(hash(path_str))

            weight_scale = 1.0
            if path_str.endswith("proj_fc"):  # from the paper: smaller init for residual projection
                weight_scale = 1.0 / jnp.sqrt(config.n_layers * 2)

            if isinstance(x, eqx.nn.Embedding | eqx.nn.Linear):
                new_weight = jr.normal(key, x.weight.shape) * 0.02 * weight_scale
                x = eqx.tree_at(lambda _x: _x.weight, x, new_weight.astype(x.weight.dtype))
                if getattr(x, "bias", None) is not None:
                    x = eqx.tree_at(lambda _x: _x.bias, x, jnp.zeros_like(x.bias))
            return x

        model = jax.tree_util.tree_map_with_path(
            _init, model, is_leaf=lambda x: isinstance(x, eqx.nn.Linear | eqx.nn.Embedding)
        )
        return model

    def lm_head(self, x):
        return x @ self.tok_embed.weight.T

    def __call__(
        self,
        idx: Int[Array, "ctx"],
        targets: Int[Array, "ctx"] | None = None,
        key: PRNGKeyArray | None = None,
    ):
        ctx_len = self.config.context_len
        input_len = idx.shape[0]

        assert targets is None or targets.shape[0] == input_len, "Input & target lengths must match"

        if input_len < ctx_len:  # pad with nans if too short
            idx_padded = jnp.pad(idx, (0, ctx_len - input_len))
        else:  # otherwise, truncate
            idx_padded = idx[-ctx_len:]

        # Embed tokens and positions
        pos = jnp.arange(ctx_len)
        x = eqx.filter_vmap(self.tok_embed)(idx_padded) + eqx.filter_vmap(self.pos_embed)(pos)
        x = x.at[input_len:].set(jnp.nan)  # mask out padding

        key = jr.PRNGKey(0) if key is None else key
        for block, bkey in zip(self.blocks, jr.split(key, len(self.blocks))):
            x = block(x, key=bkey)
        x = eqx.filter_vmap(self.final_norm)(x).astype(x.dtype)

        if targets is not None:
            logits = eqx.filter_vmap(self.lm_head)(x).astype(jnp.float32)
            labels = jax.nn.one_hot(targets, self.config.vocab_size, dtype=jnp.float32)
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.sum(jnp.nan_to_num(labels * log_probs, nan=0)) / input_len
        else:
            # Compute logits only of the requested (last) token if inference
            logits = self.lm_head(x[input_len - 1])
            loss = None

        return logits, loss

    def generate(self, idx, key, max_new_tokens=64):
        forward = eqx.nn.inference_mode(self)
        keys = jr.split(key, max_new_tokens)

        for _key in keys:
            subkey, key = jr.split(_key)
            logits, _ = eqx.filter_jit(forward)(idx, key=key)
            idx_next = jr.categorical(logits=logits, key=subkey)
            idx = jnp.concatenate([idx, jnp.array([idx_next])])
            yield idx_next
