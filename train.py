from pathlib import Path

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.extend
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax_smi import initialise_tracking
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

from helpers import WandbLogger, auto_batch_size_wrapper
from model import GPT, GPTConfig

initialise_tracking()

wandb = WandbLogger(use_wandb=(jax.process_index() == 0), name="nano_jax_gpt_test")


class TrainConfig(eqx.Module):
    batch_size: int = 0
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
    train_for: int = 5000
    dataset_name: str = "shakespear-char"


class RunConfig(eqx.Module):
    n_devices: int = 1  # number of devices available
    n_updates_on_device: int = 4  # how many steps to do without moving data to CPU

    def __init__(self):
        self.n_devices = jax.device_count()


model_config = GPTConfig()
train_config = TrainConfig()
run_config = RunConfig()

if train_config.dataset_name == "shakespear-char":
    from prepare_shakespear import decode
elif train_config.dataset_name == "openwebtext":
    from prepare_openwebtext import decode
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

    x = jnp.stack([jnp.array(data[i : i + model_config.context_len]) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1 : i + 1 + model_config.context_len]) for i in ix])

    return x.reshape((*shape, model_config.context_len)), y.reshape(
        (*shape, model_config.context_len)
    )


@eqx.filter_value_and_grad
def loss_fn(
    model: GPT,
    X: Int[Array, "devices batch ctx"],
    y: Int[Array, "devices batch ctx"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    return eqx.filter_vmap(model)(X[0], y[0], key[0])[1].mean()


def scan_fn(carry, data, model_s):
    model_d, opt_state, key = carry
    model = eqx.combine(model_d, model_s)
    X, y = data
    loss, grads = loss_fn(model, X, y, jr.split(key, X.shape[:-1]))
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return (eqx.partition(model, eqx.is_array)[0], opt_state, jr.split(key)[0]), loss


def upd_fn(model, opt_state, X, y, key):
    model_d, model_s = eqx.partition(model, eqx.is_array)
    (model_d, opt_state, _), loss = eqxi.scan(
        eqx.Partial(scan_fn, model_s=model_s), (model_d, opt_state, key), (X, y), kind="lax"
    )
    return eqx.combine(model_d, model_s), opt_state, loss.mean()


def main(batch_size=train_config.batch_size, *, exit_after_first_step=False):
    model = GPT(jr.key(0), model_config)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    xsample = get_batches("test", key=jr.key(1), shape=(1,))[0][0]
    evals_table = []

    def save(i):
        out = eqx.filter_jit(model.generate)(idx=xsample, key=jr.key(42))
        text = decode([int(x) for x in out])
        print(text)
        evals_table.append([text])
        wandb.log({"generation": wandb.Table(["text"], data=evals_table)}, commit=True)

        eqx.tree_serialise_leaves(f"checkpoints/model_{i}.eqx", model)

    for i in (pbar := tqdm(range(train_config.train_for // run_config.n_updates_on_device))):
        if i % (train_config.train_for // (run_config.n_updates_on_device * 20)) == 1:
            save(i)
        data_key, fwd_key = jr.split(jr.key(i))
        X, y = get_batches(
            "train",
            data_key,
            shape=(
                run_config.n_updates_on_device,
                run_config.n_devices,
                batch_size // run_config.n_devices,
            ),
        )
        model, opt_state, loss = eqx.filter_jit(upd_fn, donate="all")(
            model, opt_state, X, y, fwd_key
        )
        del X, y, fwd_key  # since donate="all" make sure to GC the vars too
        pbar.set_description(f"loss: {loss.mean()}")
        if exit_after_first_step:
            return
        wandb.log({"loss": loss.mean(), "step": i})
    save(train_config.train_for // run_config.n_updates_on_device)


if __name__ == "__main__":
    optim = optax.chain(
        optax.clip_by_global_norm(train_config.global_norm),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(**train_config.lr_config), weight_decay=1e-6
        ),
    )
    auto_batch_size_wrapper(main, batch_size=train_config.batch_size)
