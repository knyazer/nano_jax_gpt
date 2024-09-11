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
from jaxtyping import Array, Int, PRNGKeyArray
from tqdm import tqdm

from configs import GPTConfig, RunConfig, TrainConfig
from helpers import WandbLogger
from model import GPT

initialise_tracking()

# enable fast rng keys, unstable; current jax version: 0.4.31 (check the uv lock file)
# when I am running on TPU I install nightly, so I don't know what exact commit it will be.
# To figure it out, just look up the time of the run in the logs and check the commit at that time.
jax.config.update("jax_threefry_partitionable", True)  # noqa

# enable compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)


parser = argparse.ArgumentParser(description="Script to run with a model argument")
parser.add_argument(
    "--model",
    type=str,
    choices=["gpt2", "chargpt"],
    default="chargpt",
    help="Specify the model to use (gpt2 or chargpt). Default is chargpt.",
)
args = parser.parse_args()

wandb = WandbLogger(use_wandb=(jax.process_index() == 0), name=f"testing-{args.model}")

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


def loss_fn(
    model: GPT, X: Int[Array, "batch ctx"], y: Int[Array, "batch ctx"], key: PRNGKeyArray | None
):
    return eqx.filter_vmap(model)(X, y, key)[1].mean()


@eqx.filter_jit(donate="all")  # donate="all" allows to reuse args memory: free x2-ish memory
def step_fn(model, optim, opt_state, X, y, key, sharding):
    # shard the model and the data
    model, opt_state = eqx.filter_shard((model, opt_state), sharding.replicate())
    X, y = eqx.filter_shard((X, y), sharding)

    # accumulation function: just compute the gradient and the loss
    def grad_acc_scan_fn(key: PRNGKeyArray, data: tuple):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, *data, key=jr.split(key, data[0].shape[:-1])
        )
        return jr.split(key)[0], (loss, grads)

    # partition manually, since eqxi.scan does not do it automatically
    model_d, model_s = eqx.partition(model, eqx.is_array)
    _, (loss, grads) = eqxi.scan(grad_acc_scan_fn, key, (X, y), kind="lax")

    # compute the mean loss and grads from the accumulated values
    loss = jax.tree.map(lambda x: jnp.mean(x), loss, is_leaf=eqx.is_inexact_array)
    grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads, is_leaf=eqx.is_inexact_array)

    # step the model
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss.mean()


evals_table = []


def eval_fn(inference_model, eval_generator, batch_size, sharding):
    def evaluate(model, X, y, sharding):
        # shard the model
        model = eqx.filter_shard(model, sharding.replicate())
        X, y = eqx.filter_shard((X, y), sharding)

        # accumulate loss for requested number of batches
        _, losses = jax.lax.scan(
            eqx.Partial(lambda _, data: (None, loss_fn(model, *data, key=None))), None, (X, y)
        )
        return losses.mean()  # return the mean

    eval_x, eval_y = get_batches(
        "test", eval_generator, shape=(run_config.n_batches_in_eval, batch_size)
    )
    eval_x, eval_y = eqx.filter_shard((jnp.array(eval_x), jnp.array(eval_y)), sharding)
    eval_loss = float(eqx.filter_jit(evaluate)(inference_model, eval_x, eval_y, sharding).mean())

    # generate a continuation for some random text, and record it to wandb
    test_sample = get_batches("test", np.random.default_rng(11), shape=(1,))[0][0]
    out = eqx.filter_jit(inference_model.generate)(idx=test_sample, key=jr.key(42))
    text = decode([int(x) for x in out])
    evals_table.append([text])
    wandb.log({"text": wandb.Table(["text"], data=evals_table), "eval_loss": eval_loss})
    return eval_loss


def main():
    optim = optax.chain(
        optax.clip_by_global_norm(train_config.global_norm),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(**train_config.lr_config), weight_decay=1e-1
        ),
    )
    model = GPT(jr.key(0), model_config)
    n_model_params = jax.tree.map(lambda x: x.size, eqx.filter(model, eqx.is_array))
    n_model_params = sum(jax.tree.leaves(n_model_params))

    print(f"Model has {n_model_params/1_000_000:.2f}M parameters")
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    eval_loss = float(jnp.nan)

    # we partition the data by the batch dimension
    devices = mesh_utils.create_device_mesh((1, jax.device_count(), 1))  # grad_accum x batch x data

    # sharding stuff
    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    model = eqx.filter_shard(model, replicated)

    # random generator for numpy: cannot use jax due to synchronization
    train_generator = np.random.default_rng(42)
    eval_generator = np.random.default_rng(69)

    def checkpoint(i):
        Path("checkpoints").mkdir(exist_ok=True)  # create directory if not there
        eqx.tree_serialise_leaves(f"checkpoints/model_{i}.eqx", model)
        wandb.log_artifact("checkpoint", f"checkpoints/model_{i}.eqx")

    def load_train_batches():  # simpler function to load train batches
        return get_batches(
            "train",
            train_generator,
            shape=(
                train_config.n_grad_accumulation,
                train_config.batch_size,
            ),
        )

    X, y = load_train_batches()  # preload, so that later we can do async
    for i in (pbar := tqdm(range(train_config.train_for))):
        data_key, fwd_key = jr.split(jr.key(i))

        t = time.time()

        X, y = eqx.filter_shard((jnp.array(X), jnp.array(y)), sharding)  # shard preloaded data
        model, opt_state, loss = step_fn(model, optim, opt_state, X, y, fwd_key, sharding)
        X, y = load_train_batches()  # async load
        loss = float(loss.mean())  # wait for jax to sync

        # log
        pbar.set_description(
            f"loss:{loss:.2f} / eval:{eval_loss:.2f} | step:{(time.time() - t)*1e3:.2f}ms"
        )
        wandb.log({"loss": loss})

        # if we want to checkpoint..
        chckp_freq = train_config.train_for // run_config.times_to_checkpoint
        if i % chckp_freq == chckp_freq - 1:
            checkpoint(i)

        # if we want to eval...
        eval_freq = train_config.train_for // run_config.times_to_eval
        if i % eval_freq == eval_freq - 1:
            eval_loss = eval_fn(
                eqx.nn.inference_mode(model), eval_generator, train_config.batch_size, sharding
            )

    # checkpoint the final model
    checkpoint(train_config.train_for)


if __name__ == "__main__":
    main()
