from pathlib import Path

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax_smi import initialise_tracking
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

from model import GPT, GPTConfig
from prepare_shakespear import decode

initialise_tracking()


class TrainConfig(eqx.Module):
    batch_size: int = 64
    lr_config: dict = eqx.field(
        default_factory=lambda: {
            "init_value": 1e-3,
            "peak_value": 1e-3,
            "warmup_steps": 100,
            "decay_steps": 5_000,
            "end_value": 1e-4,
        }
    )
    global_norm: float = 1.0
    train_for: int = 5_000


model_config = GPTConfig()
train_config = TrainConfig()


def get_batches(split: str, key: PRNGKeyArray, n: int):
    data_dir = Path()
    if split == "train":
        data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    else:
        data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    ix = jr.randint(
        key=key,
        minval=0,
        maxval=len(data) - model_config.context_len,
        shape=(train_config.batch_size * n,),
    )

    x = jnp.stack([jnp.array(data[i : i + model_config.context_len]) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1 : i + 1 + model_config.context_len]) for i in ix])
    target_shape = (n, train_config.batch_size, model_config.context_len)

    return x.reshape(target_shape), y.reshape(target_shape)


@eqx.filter_value_and_grad
def loss_fn(
    model: GPT, X: Int[Array, "batch ctx"], y: Int[Array, "batch ctx"], key: PRNGKeyArray
) -> Float[Array, ""]:
    return eqx.filter_vmap(model)(X, y, key)[1].mean()


if __name__ == "__main__":
    model = GPT(jr.key(0), model_config)
    optim = optax.chain(
        optax.clip_by_global_norm(train_config.global_norm),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(**train_config.lr_config), weight_decay=1e-6
        ),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def scan_fn(carry, data, model_s):
        model_d, opt_state, key = carry
        model = eqx.combine(model_d, model_s)
        X, y = data
        loss, grads = loss_fn(model, X, y, jr.split(key, train_config.batch_size))
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return (eqx.partition(model, eqx.is_array)[0], opt_state, jr.split(key)[0]), loss

    @eqx.filter_jit
    def upd_fn(model, opt_state, X, y, key):
        model_d, model_s = eqx.partition(model, eqx.is_array)
        (model_d, opt_state, _), loss = eqxi.scan(
            eqx.Partial(scan_fn, model_s=model_s), (model_d, opt_state, key), (X, y), kind="lax"
        )
        return eqx.combine(model_d, model_s), opt_state, loss.mean()

    xsample = get_batches("test", key=jr.key(0), n=1)[0][0][0]
    for i in (pbar := tqdm(range(train_config.train_for // 10))):
        if i % 10 == 0:
            out = eqx.filter_jit(model.generate)(idx=xsample, key=jr.key(42))
            print(decode([int(x) for x in out]))
        data_key, fwd_key = jr.split(jr.key(i))
        X, y = get_batches("train", data_key, n=10)
        model, opt_state, loss = upd_fn(model, opt_state, X, y, fwd_key)
        pbar.set_description(f"loss: {loss.mean()}")
