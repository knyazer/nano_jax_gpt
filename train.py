import argparse
import time
from pathlib import Path

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.extend
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import numpy as np
import optax
from jax.experimental import mesh_utils
from jax_smi import initialise_tracking
from jaxtyping import Array, Int
from tqdm import tqdm

from configs import GPTConfig, RunConfig, TrainConfig
from helpers import WandbLogger, auto_batch_size_wrapper
from model import GPT

initialise_tracking()

# enable compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

wandb = WandbLogger(use_wandb=(jax.process_index() == 0), name="nano_jax_gpt_test_2")

parser = argparse.ArgumentParser(description="Script to run with a model argument")
parser.add_argument(
    "--model",
    type=str,
    choices=["gpt2", "chargpt"],
    default="chargpt",
    help="Specify the model to use (gpt2 or chargpt). Default is chargpt.",
)
args = parser.parse_args()


model_config: GPTConfig = GPTConfig.from_preset(args.model)
train_config: TrainConfig = TrainConfig.from_preset(args.model)
run_config: RunConfig = RunConfig.from_preset(args.model)

if train_config.dataset_name == "shakespear-char":
    from prepare_shakespear import decode  # type: ignore
elif train_config.dataset_name == "openwebtext":
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    def decode(s):
        return enc.decode(s)
else:
    raise AssertionError("Unknown dataset")


def get_batches(split: str, rng: np.random.Generator, shape: tuple):
    data_dir = Path()
    if split == "train":
        data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    else:
        data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    ix = rng.integers(
        low=0,
        high=len(data) - model_config.context_len - 1,
        size=(int(np.prod(shape)),),
    )

    x = np.stack([np.array(data[i : i + model_config.context_len]) for i in ix])
    y = np.stack([np.array(data[i + 1 : i + 1 + model_config.context_len]) for i in ix])

    return x.reshape((*shape, model_config.context_len)), y.reshape(
        (*shape, model_config.context_len)
    )


@eqx.filter_value_and_grad
def loss_fn(model, X: Int[Array, "batch ctx"], y: Int[Array, "batch ctx"], key):
    return eqx.filter_vmap(model)(X, y, key)[1].mean()


def scan_fn(carry, data, model_s):
    model_d, opt_state, key = carry
    model = eqx.combine(model_d, model_s)
    X, y = data
    loss, grads = loss_fn(model, X, y, jr.split(key, X.shape[:-1]))
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return (eqx.partition(model, eqx.is_array)[0], opt_state, jr.split(key)[0]), loss


def upd_fn(model, opt_state, X, y, key, sharding):
    model, opt_state = eqx.filter_shard((model, opt_state), sharding.replicate())
    X, y = eqx.filter_shard((X, y), sharding)

    model_d, model_s = eqx.partition(model, eqx.is_array)
    (model_d, opt_state, _), loss = jax.lax.scan(
        eqx.Partial(scan_fn, model_s=model_s), (model_d, opt_state, key), (X, y)
    )
    return eqx.combine(model_d, model_s), opt_state, loss.mean()


def eval_scan_fn(_, data, model):
    losses = eqx.filter_vmap(model)(*data)[1].mean()
    return None, losses


def evaluate(model, X, y, sharding):
    model = eqx.filter_shard(model, sharding.replicate())
    X, y = eqx.filter_shard((X, y), sharding)
    _, losses = eqxi.scan(eqx.Partial(eval_scan_fn, model=model), None, (X, y), kind="lax")
    return losses.mean()


evals_table = []


def eval_fn(model, eval_generator, batch_size, sharding):
    eval_x, eval_y = get_batches(
        "test", eval_generator, shape=(run_config.n_batches_in_eval, batch_size)
    )
    eval_x, eval_y = eqx.filter_shard((jnp.array(eval_x), jnp.array(eval_y)), sharding)
    eval_loss = float(
        eqx.filter_jit(evaluate)(eqx.nn.inference_mode(model), eval_x, eval_y, sharding).mean()
    )

    test_sample = get_batches("test", np.random.default_rng(11), shape=(1,))[0][0]
    out = eqx.filter_jit(model.generate)(idx=test_sample, key=jr.key(42))
    text = decode([int(x) for x in out])
    evals_table.append([text])
    wandb.log({"text": wandb.Table(["text"], data=evals_table), "eval_loss": eval_loss})
    return eval_loss


def main(batch_size=train_config.batch_size, *, exit_after_first_step=False):
    model = GPT(jr.key(0), model_config)
    n_model_params = jax.tree.map(lambda x: x.size, eqx.filter(model, eqx.is_array))
    n_model_params = sum(jax.tree_leaves(n_model_params))
    print(f"Model has {n_model_params/1_000_000:.2f}M parameters")
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    eval_loss = float(jnp.nan)

    devices = mesh_utils.create_device_mesh((1, jax.device_count(), 1))  # reps x batch x data
    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    model = eqx.filter_shard(model, replicated)

    def checkpoint(i):
        Path("checkpoints").mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(f"checkpoints/model_{i}.eqx", model)

        wandb.log_artifact("chk", f"checkpoints/models_{i}.eqx")

    n_steps = train_config.train_for // run_config.n_updates_on_device

    train_generator = np.random.default_rng(42)
    eval_generator = np.random.default_rng(69)

    def load_train_batches():
        return get_batches(
            "train",
            train_generator,
            shape=(
                run_config.n_updates_on_device,
                batch_size,
            ),
        )

    X, y = load_train_batches()
    for i in (pbar := tqdm(range(n_steps))):
        data_key, fwd_key = jr.split(jr.key(i))

        t = time.time()

        X, y = eqx.filter_shard((jnp.array(X), jnp.array(y)), sharding)
        model, opt_state, loss = eqx.filter_jit(upd_fn, donate="all")(
            model, opt_state, X, y, fwd_key, sharding
        )
        X, y = load_train_batches()  # async load
        loss = float(loss.mean())  # wait for jax to finish
        step_time = (time.time() - t) / run_config.n_updates_on_device

        # log
        pbar.set_description(f"loss:{loss:.2f} / eval:{eval_loss:.2f} | step:{step_time*1e3:.2f}ms")
        if exit_after_first_step:
            return
        wandb.log({"loss": loss})

        # if we want to checkpoint..
        chckp_freq = n_steps // run_config.times_to_checkpoint
        if i % chckp_freq == chckp_freq - 1:
            checkpoint(i)

        # if we want to eval...
        eval_freq = n_steps // run_config.times_to_eval
        if i % eval_freq == eval_freq - 1:
            eval_loss = eval_fn(model, eval_generator, batch_size, sharding)

    checkpoint(train_config.train_for // run_config.n_updates_on_device)


if __name__ == "__main__":
    optim = optax.chain(
        optax.clip_by_global_norm(train_config.global_norm),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(**train_config.lr_config), weight_decay=1e-2
        ),
    )
    auto_batch_size_wrapper(
        main, batch_size=train_config.batch_size, n_devices=run_config.n_devices
    )
