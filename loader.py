import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import tiktoken
from transformers import FlaxGPT2LMHeadModel

from attn import FlashMultiheadAttention
from model import GPT, GPTConfig


def linear_from_pretrained(hf_linear, *, transpose=False):
    kernel = hf_linear["kernel"].T if transpose else hf_linear["kernel"]
    bias = hf_linear["bias"].T if transpose else hf_linear["bias"]
    output_features, input_features = kernel.shape
    layer = eqx.nn.Linear(input_features, output_features, key=jr.key(0))
    layer = eqx.tree_at(lambda t: t.weight, layer, kernel)
    layer = eqx.tree_at(lambda t: t.bias, layer, bias)
    return layer


def norm_from_pretrained(hf_norm):
    norm = eqx.nn.LayerNorm(hf_norm["scale"].shape, eps=1e-5)
    norm = eqx.tree_at(lambda t: t.weight, norm, hf_norm["scale"])
    norm = eqx.tree_at(lambda t: t.bias, norm, hf_norm["bias"])
    return norm


def _qkv_split(qkv_layer):
    weights = qkv_layer.weight
    biases = qkv_layer.bias

    # openai uses conv1d instead of linear sometimes, and we need to transpose such weights
    w1, w2, w3 = jnp.split(weights.T, 3, axis=0)
    b1, b2, b3 = jnp.split(biases.T, 3, axis=0)

    res = []
    for w, b in zip((w1, w2, w3), (b1, b2, b3)):
        layer = eqx.nn.Linear(w.shape[0], w.shape[1], key=jr.key(0))
        layer = eqx.tree_at(lambda t: t.weight, layer, w)
        layer = eqx.tree_at(lambda t: t.bias, layer, b)
        res.append(layer)
    return res


def attn_from_pretrained(hf_attn, config: GPTConfig):
    attn = FlashMultiheadAttention(
        num_heads=config.n_heads,
        query_size=config.n_embed,
        dropout_p=config.dropout,
        key=jr.key(0),
    )

    # openai implementation uses a single layer for qkv instead of three separate ones,
    # so we have to manually partition the params
    qkv_expand = linear_from_pretrained(hf_attn["c_attn"], transpose=True)
    q, k, v = _qkv_split(qkv_expand)
    attn = eqx.tree_at(lambda t: t.query_proj, attn, q)
    attn = eqx.tree_at(lambda t: t.key_proj, attn, k)
    attn = eqx.tree_at(lambda t: t.value_proj, attn, v)

    attn = eqx.tree_at(lambda t: t.output_proj, attn, linear_from_pretrained(hf_attn["c_proj"]))
    return attn


def block_from_pretrained(block, hf_block, config: GPTConfig):
    # attn
    block = eqx.tree_at(lambda b: b.lnorm_attn, block, norm_from_pretrained(hf_block["ln_1"]))
    block = eqx.tree_at(lambda b: b.attn, block, attn_from_pretrained(hf_block["attn"], config))

    # mlp
    block = eqx.tree_at(lambda b: b.lnorm_mlp, block, norm_from_pretrained(hf_block["ln_2"]))
    new_expand = linear_from_pretrained(hf_block["mlp"]["c_fc"])
    new_proj = linear_from_pretrained(hf_block["mlp"]["c_proj"])
    block = eqx.tree_at(lambda b: b.expand_fc, block, new_expand)
    block = eqx.tree_at(lambda b: b.proj_fc, block, new_proj)
    return block


def gpt_from_pretrained():
    # some notes: we support only gpt2 for now, since this is the most common use case
    config = GPTConfig().from_preset("gpt2")
    model = GPT(jr.PRNGKey(0), config)
    dyn_model, st_model = eqx.partition(model, eqx.is_array)

    # lets ensure we override all the weights, by setting the default to nan
    model = jax.tree.map(lambda x: jnp.nan * x if eqx.is_array(x) else x, model)

    # load the pretrained model, we use flax since it's the closest it gets to Equinox
    model_hf = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
    params_hf = model_hf._params["transformer"]  # noqa # type: ignore

    # a funny bug here: you cannot just iterate over params_hf['h'].values() cuz
    # the order will be wrong; don't ask me how much time I spent debugging this lol
    block_list = [
        block_from_pretrained(ours, params_hf["h"][str(ind)], config)
        for ind, ours in enumerate(model.blocks)
    ]
    model = eqx.tree_at(lambda m: m.blocks, model, block_list)
    model = eqx.tree_at(lambda m: m.tok_embed.weight, model, params_hf["wte"]["embedding"])
    model = eqx.tree_at(lambda m: m.pos_embed.weight, model, params_hf["wpe"]["embedding"])
    model = eqx.tree_at(lambda m: m.final_norm, model, norm_from_pretrained(params_hf["ln_f"]))

    # make sure all static things are the same as before
    new_dyn_model, new_st_model = eqx.partition(model, eqx.is_array)
    shapes = jax.tree.map(
        lambda x, y: x.shape == y.shape, dyn_model, new_dyn_model, is_leaf=eqx.is_array
    )
    values = jax.tree.map(
        lambda x, y: x == y,
        st_model,
        new_st_model,
        is_leaf=lambda leaf: type(leaf) in [int, float, tuple, bool],
    )
    assert all(
        jax.tree.leaves(shapes)
    ), "Inconsistent shapes between the loaded and original models!"
    assert all(
        jax.tree.leaves(values)
    ), "Different static values between the loaded and original models!"

    return model


if __name__ == "__main__":
    model = gpt_from_pretrained()
    enc = tiktoken.get_encoding("gpt2")

    text = "My name is Thomas and my main"  # example from HuggingFace
    ids = jnp.array(enc.encode(text))

    print(text, end="", flush=True)
    for _ in range(20):
        out, _ = model(ids)
        tok = jnp.argmax(out)  # greedy decoding: temp is 0
        ids = jnp.concatenate([ids, tok[None]])
        print(enc.decode([tok]), end="", flush=True)  # type: ignore

    assert enc.decode([int(x) for x in ids]) == (
        "My name is Thomas and my main goal is to make sure that "
        "I'm not just a guy who's going to be a part of"
    ), "The produced sequence is not exactly the same as the one from HF!"
