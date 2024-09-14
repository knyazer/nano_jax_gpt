import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import tiktoken
from jax.lax import transpose_p
from transformers import FlaxGPT2LMHeadModel

from attn import FlashMultiheadAttention
from model import GPT, GPTConfig


def linear_from_pretrained(hf_linear, *, transpose=False):
    kernel = hf_linear["kernel"].T if transpose else hf_linear["kernel"]
    bias = hf_linear["bias"].T if transpose else hf_linear["bias"]
    input_features, output_features = kernel.shape
    layer = eqx.nn.Linear(input_features, output_features, key=jr.key(0))
    layer = eqx.tree_at(lambda t: t.weight, layer, kernel)
    layer = eqx.tree_at(lambda t: t.bias, layer, bias)
    return layer


def norm_from_pretrained(hf_norm):
    norm = eqx.nn.LayerNorm(hf_norm["scale"].shape)
    norm = eqx.tree_at(lambda t: t.weight, norm, hf_norm["scale"])
    norm = eqx.tree_at(lambda t: t.bias, norm, hf_norm["bias"])
    return norm


def _qkv_split(qkv_layer):
    weights = qkv_layer.weight
    biases = qkv_layer.bias

    w1, w2, w3 = jnp.split(weights, 3, axis=-1)
    b1, b2, b3 = jnp.split(biases, 3, axis=-1)

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
    assert attn.query_proj.weight.shape == q.weight.shape
    assert attn.key_proj.weight.shape == k.weight.shape
    assert attn.value_proj.weight.shape == v.weight.shape
    attn = eqx.tree_at(lambda t: t.query_proj, attn, q)
    attn = eqx.tree_at(lambda t: t.key_proj, attn, k)
    attn = eqx.tree_at(lambda t: t.value_proj, attn, v)

    # output is just one-to-one
    attn = eqx.tree_at(lambda t: t.output_proj, attn, linear_from_pretrained(hf_attn["c_proj"]))
    return attn


def block_from_pretrained(block, hf_block, config: GPTConfig):
    block = eqx.tree_at(lambda b: b.lnorm_attn, block, norm_from_pretrained(hf_block["ln_1"]))
    block = eqx.tree_at(lambda b: b.lnorm_mlp, block, norm_from_pretrained(hf_block["ln_2"]))

    new_expand = linear_from_pretrained(hf_block["mlp"]["c_fc"])
    new_proj = linear_from_pretrained(hf_block["mlp"]["c_proj"])
    assert new_expand.weight.shape == block.expand_fc.weight.shape
    assert new_proj.weight.shape == block.proj_fc.weight.shape
    block = eqx.tree_at(lambda b: b.expand_fc, block, new_expand)
    block = eqx.tree_at(lambda b: b.proj_fc, block, new_proj)

    block = eqx.tree_at(lambda b: b.attn, block, attn_from_pretrained(hf_block["attn"], config))
    return block


def gpt_from_pretrained():
    config = GPTConfig().from_preset("gpt2")
    model = GPT(jr.PRNGKey(0), config)
    model = jax.tree.map(
        lambda x: jnp.nan * x if eqx.is_array(x) else x, model
    )  # make sure all arrays are nans
    model_hf = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
    params_hf = model_hf._params["transformer"]  # noqa # type: ignore

    # now, fun: exporting the weights into our model
    block_list = [
        block_from_pretrained(ours, hfs, config)
        for ours, hfs in zip(model.blocks, params_hf["h"].values())
    ]
    model = eqx.tree_at(lambda m: m.blocks, model, block_list)
    model = eqx.tree_at(lambda m: m.tok_embed.weight, model, params_hf["wte"]["embedding"])
    model = eqx.tree_at(lambda m: m.pos_embed.weight, model, params_hf["wpe"]["embedding"])
    model = eqx.tree_at(lambda m: m.final_norm, model, norm_from_pretrained(params_hf["ln_f"]))
    return model


if __name__ == "__main__":
    model = gpt_from_pretrained()
    enc = tiktoken.get_encoding("gpt2")

    text = "I love oranges."
    ids = jnp.array(enc.encode(text))
    true_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
    out = true_model(ids[None, :])
    my_out = model(ids)
    print(out, my_out)
    nids = ids
    nids = ids
    for _ in range(10):
        out = true_model(nids[None, :])
        tok = jnp.argmax(out.logits[0][-1])
        print(enc.decode([tok]))
        nids = jnp.concatenate([nids, tok[None]])
        print(enc.decode([int(x) for x in nids]))
    for _ in range(10):
        out = model(nids)
        tok = jnp.argmax(out[0])
        print(enc.decode([tok]))
        nids = jnp.concatenate([nids, tok[None]])
        print(enc.decode([int(x) for x in nids]))
