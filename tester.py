import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import termcolor
from jaxtyping import PRNGKeyArray

from configs import GPTConfig, TrainConfig
from model import GPT

jax.config.update("jax_platform_name", "cpu")

model_config = GPTConfig().from_preset("chargpt")
train_config = TrainConfig().from_preset("chargpt")
model: GPT = eqx.tree_deserialise_leaves(
    "checkpoints/model_12500.eqx", like=GPT(jr.key(1), model_config)
)

if train_config.dataset_name == "shakespear-char":
    from prepare_shakespear import decode  # type: ignore
elif train_config.dataset_name == "openwebtext":
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    def decode(s):
        return enc.decode(s)
else:
    raise AssertionError("Unknown dataset")


def get_batches(split: str, key: PRNGKeyArray, shape: tuple):
    data_dir = Path()
    if split == "train":
        data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    else:
        data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    ix = jr.randint(
        key=key,
        minval=0,
        maxval=len(data) - model_config.context_len,
        shape=(int(np.prod(shape)),),
    )

    x = np.stack([np.array(data[i : i + model_config.context_len]) for i in ix])
    y = np.stack([np.array(data[i + 1 : i + 1 + model_config.context_len]) for i in ix])

    return x.reshape((*shape, model_config.context_len)), y.reshape(
        (*shape, model_config.context_len)
    )


test_sample = get_batches("test", key=jr.key(1), shape=(1,))[0][0]
model.generate(idx=test_sample, key=jr.key(0), max_new_tokens=32)  # compile
colors = ["green", "cyan", "blue", "magenta", "yellow", "red"]

t = time.time()
for token, index in model.generate(idx=test_sample, key=jr.key(43), max_new_tokens=512):
    color = colors[min(int(index), 5)]
    print(termcolor.colored(decode([int(token)]), color), end="", flush=True)  # type: ignore

for i in range(6):
    print(f"\n\n---  took {time.time() - t}s  ---\n\n")
    t = time.time()
    for token, index in model.generate(
        idx=test_sample, key=jr.key(43), max_new_tokens=512, shortcircuit=i
    ):
        color = colors[min(int(index), 5)]
        print(termcolor.colored(decode([int(token)]), color), end="", flush=True)  # type: ignore

print(f"\n\n---  took {time.time() - t}s  ---\n\n")


for T in [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]:
    t = time.time()

    def p_wrap(p):
        alpha = p * T
        if alpha < 0.05:
            return alpha + 0.5 * alpha * (1 - alpha) * T
        return 1.0 - jnp.exp(-alpha)

    for token, index in model.generate(
        idx=test_sample, key=jr.key(40), max_new_tokens=512, p_wrap=p_wrap
    ):
        color = colors[min(int(index), 5)]
        print(termcolor.colored(decode([int(token)]), color), end="", flush=True)  # type: ignore

    print(f"\n\n---  took {time.time() - t}s  ---\n\n")
