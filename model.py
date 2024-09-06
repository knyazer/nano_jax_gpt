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
    conditional_limit: float

    def __init__(self, config: GPTConfig, key: PRNGKeyArray, p_size: int = 10):
        k1, k2 = jr.split(key)
        self.linear = eqx.nn.Linear(
            config.n_embed, config.vocab_size + p_size, use_bias=False, key=k1
        )
        self.lnorm = eqx.nn.RMSNorm(config.n_embed, use_bias=False)
        self.p_linear = eqx.nn.Linear(p_size, 1, key=k2)
        self.p_size = p_size
        self.conditional_limit = config.conditional_limit

    def __call__(self, x: Float[Array, "embed"]):
        out = self.linear(self.lnorm(x))
        p = self.p_linear(jax.nn.gelu(out[: self.p_size], approximate=True))
        return jax.nn.sigmoid(p) * self.conditional_limit, out[self.p_size :]


class GPTLog(eqx.Module):
    accept_rates: Float[Array, "n_layer"]
    layerwise_losses: Float[Array, "n_layer"]
    layerwise_weighted: Float[Array, "n_layer"]
    expected_compute: Float[Array, ""]

    def __init__(self, accept_rates, layerwise_losses, layerwise_weighted, expected_compute):
        self.accept_rates = jnp.array(accept_rates)
        self.layerwise_losses = jnp.array(layerwise_losses)
        self.expected_compute = jnp.array(expected_compute)
        self.layerwise_weighted = jnp.array(layerwise_weighted)


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
            MiddleHead(config, key=dec_key) for dec_key in jr.split(k5, config.n_layer - 1)
        ]
        self.config = config

    def __call__(
        self,
        idx: Int[Array, "ctx"],
        targets: Int[Array, "ctx"] | None = None,
        key: PRNGKeyArray | None = None,
        *,
        generation: bool = False,
    ):
        labels = jax.nn.one_hot(targets, self.config.vocab_size) if targets is not None else None

        x = eqx.filter_vmap(self.tok_embed)(idx)
        if key is None:
            key = jr.PRNGKey(0)  # inference, i guess, so dummy key
        keys = jr.split(key, len(self.blocks))

        already_accepted_p = jnp.zeros((self.config.context_len,), dtype=jnp.float32)
        total_loss = jnp.zeros((self.config.context_len,), dtype=jnp.float32)
        current_estimated_compute = 4.0

        accept_rates = []
        layerwise_losses = []
        layerwise_weighted = []
        expected_compute = 0.0
        unscaled_aap = 0.0

        def combine_compute_and_loss(compute, loss):
            return compute * loss

        for block_index, (block, bkey) in enumerate(zip(self.blocks, keys)):  # todo: jax.lax.scan
            p_key, _bkey = jr.split(bkey)
            x = eqx.filter_jit(block)(x, key=_bkey)
            if generation and block_index != len(self.blocks) - 1:
                accept_p, logit = self.middle_heads[block_index](x[-1])
                # sample from block_accept_p
                if jr.bernoulli(p_key, accept_p).ravel()[0]:
                    return logit, None, None
            else:
                current_estimated_compute += 1
                if labels is not None and block_index != len(self.blocks) - 1:
                    block_accept_p, logits = eqx.filter_vmap(self.middle_heads[block_index])(x)
                    block_accept_p = block_accept_p.ravel()
                    # we regularize by forcing geom mean block accept to be at most conditional limi
                    log_probs = jax.nn.log_softmax(logits)
                    loss = -jnp.sum(labels * log_probs, axis=1) / self.config.context_len
                    joint_block_p = (1.0 - already_accepted_p) * block_accept_p
                    already_accepted_p += joint_block_p
                    unscaled_aap += joint_block_p.mean()
                    expected_compute += joint_block_p.mean() * current_estimated_compute
                    # log(compute) * loss = C
                    total_loss += joint_block_p * combine_compute_and_loss(
                        current_estimated_compute, loss
                    )

                    layerwise_losses.append(loss.sum())
                    layerwise_weighted.append((joint_block_p * loss).sum() / joint_block_p.sum())
                    accept_rates.append(joint_block_p.mean())

        # total loss is expectation of the logit diff wrt
        x = eqx.filter_vmap(self.final_norm)(x)
        if not generation:
            logits = eqx.filter_vmap(self.lm_head)(x)
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.sum(labels * log_probs, axis=1) / self.config.context_len
            joint_accept_p = 1.0 - already_accepted_p
            total_loss += joint_accept_p * combine_compute_and_loss(current_estimated_compute, loss)
            expected_compute += (1.0 - unscaled_aap) * current_estimated_compute

            layerwise_losses.append(loss.sum())
            layerwise_weighted.append((joint_accept_p * loss).sum() / joint_accept_p.sum())
            accept_rates.append(joint_accept_p.mean())
        else:
            logits = self.lm_head(x[-1])
        return (
            logits,
            total_loss.sum() / current_estimated_compute,
            GPTLog(accept_rates, layerwise_losses, layerwise_weighted, expected_compute - 4.0),
        )

    def generate(self, idx, key, max_new_tokens=100):
        idx = idx.astype(jnp.int32)
        forward = eqx.nn.inference_mode(self)
        new_stuff = []

        for _ in range(max_new_tokens):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            logits, *_ = forward(idx, generation=True, key=subkey1)
            idx_next = jax.random.categorical(logits=logits, key=subkey2)

            idx = jnp.roll(idx, -1)
            idx = idx.at[-1].set(idx_next)
            new_stuff.append(idx_next)

        return jnp.array(new_stuff).ravel()
