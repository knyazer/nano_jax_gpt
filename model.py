import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from configs import GPTConfig


class Block(eqx.Module):
    expand_fc: eqx.nn.Linear
    proj_fc: eqx.nn.Linear
    lnorm_attn: eqx.nn.RMSNorm
    lnorm_mlp: eqx.nn.RMSNorm
    attn: eqx.nn.MultiheadAttention
    rope: eqx.nn.RotaryPositionalEmbedding
    dropout: eqx.nn.Dropout

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3 = jr.split(key, 3)
        self.expand_fc = eqx.nn.Linear(config.n_embed, 4 * config.n_embed, use_bias=False, key=k1)
        self.proj_fc = eqx.nn.Linear(config.n_embed * 4, config.n_embed, use_bias=False, key=k2)
        self.lnorm_attn = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.lnorm_mlp = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.attn = eqx.nn.MultiheadAttention(
            config.n_head, config.n_embed, dropout_p=config.dropout, key=k3
        )
        self.rope = eqx.nn.RotaryPositionalEmbedding(config.n_embed, 10_000)
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(
        self, x: Float[Array, "ctx emb"], key: PRNGKeyArray | None = None
    ) -> Float[Array, "ctx emb"]:
        if key is None:
            mlp_key, attn_key = None, None
        else:
            mlp_key, attn_key = jr.split(key)
        x = eqx.filter_vmap(self.lnorm_attn)(x)
        x = x + self.attn(
            query=self.rope(x),
            key_=self.rope(x),
            value=x,
            mask=jnp.tril(jnp.ones((x.shape[0], x.shape[0]))),
            key=attn_key,
        )
        x = x + self.dropout(
            eqx.filter_vmap(
                lambda tok: self.lnorm_mlp(
                    self.proj_fc(jax.nn.gelu(self.expand_fc(tok), approximate=True))
                )
            )(x),
            key=mlp_key,
        )
        return x


class MiddleHead(eqx.Module):
    linear: eqx.nn.Linear
    p_linear: eqx.nn.Linear
    p_size: int
    lnorm: eqx.nn.RMSNorm

    def __init__(self, embed_size: int, vocab_size: int, key: PRNGKeyArray, p_size: int = 10):
        k1, k2 = jr.split(key)
        self.linear = eqx.nn.Linear(embed_size, vocab_size + p_size, use_bias=False, key=k1)
        self.lnorm = eqx.nn.RMSNorm(embed_size, use_bias=False)
        p_linear = eqx.nn.Linear(p_size, 1, key=k2)
        self.p_linear = eqx.tree_at(
            lambda layer: layer.bias,
            p_linear,
            p_linear.bias - 10.0,  # type: ignore
        )
        self.p_size = p_size

    def __call__(self, x: Float[Array, "embed"]):
        out = self.linear(self.lnorm(x))
        p = self.p_linear(jax.nn.gelu(out[: self.p_size], approximate=True))
        return jnp.min(jnp.stack((jax.nn.sigmoid(p), jnp.array([0.7])))), out[self.p_size :]


class GPT(eqx.Module):
    config: GPTConfig = eqx.field(static=True)
    tok_embed: eqx.nn.Embedding
    blocks: list[Block]
    final_norm: eqx.nn.RMSNorm
    lm_head: eqx.nn.Linear
    middle_heads: list[MiddleHead]

    def __init__(self, key: PRNGKeyArray, config: GPTConfig):
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        self.tok_embed = eqx.nn.Embedding(config.vocab_size, config.n_embed, key=k2)
        self.blocks = [Block(block_key, config) for block_key in jr.split(k3, config.n_layer)]
        self.final_norm = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.lm_head = eqx.nn.Linear(config.n_embed, config.vocab_size, use_bias=False, key=k4)
        self.middle_heads = [
            MiddleHead(config.n_embed, config.vocab_size, key=dec_key)
            for dec_key in jr.split(k5, config.n_layer - 1)
        ]
        self.config = config

    def __call__(
        self,
        idx: Int[Array, "ctx"],
        targets: Int[Array, "ctx"] | None = None,
        key: PRNGKeyArray | None = None,
        *,
        evaluation: bool = False,
    ):
        labels = jax.nn.one_hot(targets, self.config.vocab_size) if targets is not None else None

        x = eqx.filter_vmap(self.tok_embed)(idx)
        if key is None:
            key = jr.PRNGKey(0)  # inference, i guess, so dummy key
        keys = jr.split(key, len(self.blocks))

        already_accepted_p = jnp.zeros((self.config.context_len,), dtype=jnp.float32)
        total_loss = jnp.zeros((self.config.context_len,), dtype=jnp.float32)
        current_estimated_compute = 4
        accepts_log = []

        for block_index, (block, bkey) in enumerate(zip(self.blocks, keys)):  # todo: jax.lax.scan
            x = block(x, key=bkey)
            current_estimated_compute += 1
            if labels is not None and block_index != len(self.blocks) - 1 and not evaluation:
                block_accept_p, logits = eqx.filter_vmap(self.middle_heads[block_index])(x)
                block_accept_p = block_accept_p.ravel()
                log_probs = jax.nn.log_softmax(logits)
                loss = -jnp.sum(labels * log_probs, axis=1) / self.config.context_len
                joint_block_p = (1.0 - already_accepted_p) * block_accept_p
                accepts_log.append(joint_block_p.mean())
                already_accepted_p += joint_block_p
                total_loss += joint_block_p * current_estimated_compute * loss

        # total loss is expectation of the logit diff wrt
        x = eqx.filter_vmap(self.final_norm)(x)
        if labels is not None:
            logits = eqx.filter_vmap(self.lm_head)(x)
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.sum(labels * log_probs, axis=1) / self.config.context_len
            total_loss += (1.0 - already_accepted_p) * loss * current_estimated_compute
        else:
            logits = self.lm_head(x[-1])
        final_loss = total_loss.sum() / current_estimated_compute
        return logits, final_loss, jnp.array(accepts_log)

    def generate(self, idx, key, max_new_tokens=256):
        forward = eqx.nn.inference_mode(self)

        def scan_fn(carry, _):
            idx, key = carry
            key, subkey = jax.random.split(key)

            logits, *_ = forward(idx)
            idx_next = jr.categorical(logits=logits, key=subkey)

            idx = jnp.roll(idx, -1)
            idx = idx.at[-1].set(idx_next)
            return (idx, key), idx_next

        _, new_stuff = jax.lax.scan(scan_fn, (idx, key), None, length=max_new_tokens)
        return new_stuff.ravel()
