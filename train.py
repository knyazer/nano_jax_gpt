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
from jax.experimental import mesh_utils
from jax_smi import initialise_tracking
from jaxtyping import Array, Int, PRNGKeyArray
from tqdm import tqdm
from transformers.generation.flax_utils import Any

from configs import GPTConfig, RunConfig, TrainConfig
from helpers import WandbLogger
from model import GPT

initialise_tracking()

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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
    from prepare_shakespear import decode
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
    low_p_model = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x,
        model,
        is_leaf=eqx.is_inexact_array,
    )
    return eqx.filter_vmap(low_p_model)(X, y, key)[1].astype(jnp.float32).mean()


@eqx.filter_jit(donate="all")
def step_fn(model, optim, opt_state, X, y, key):
    def grad_acc_scan_fn(key: PRNGKeyArray, data: tuple):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, *data, key=jr.split(key, data[0].shape[:-1])
        )
        return jr.split(key)[0], (loss, grads)

    _, (loss, grads) = eqxi.scan(grad_acc_scan_fn, key, (X, y), kind="lax")

    loss = jax.tree.map(lambda x: jnp.mean(x), loss, is_leaf=eqx.is_inexact_array_like)
    grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads, is_leaf=eqx.is_inexact_array_like)

    return model, opt_state, loss.mean()


evals_table = []


def eval_fn(inference_model, eval_generator, batch_size, sharding):
    def evaluate(model, X, y, sharding):
        model = eqx.filter_shard(model, sharding.replicate())
        X, y = eqx.filter_shard((X, y), sharding)

        _, losses = jax.lax.scan(
            eqx.Partial(lambda _, data: (None, loss_fn(model, *data, key=None))), None, (X, y)
        )
        return losses.astype(jnp.float32).mean()

    eval_x, eval_y = get_batches(
        "test", eval_generator, shape=(run_config.n_batches_in_eval, batch_size)
    )
    eval_x, eval_y = eqx.filter_shard((jnp.array(eval_x), jnp.array(eval_y)), sharding)
    eval_loss = float(eqx.filter_jit(evaluate)(inference_model, eval_x, eval_y, sharding).mean())

    if False:
        test_sample = get_batches("test", np.random.default_rng(11), shape=(1,))[0][0]
        out = inference_model.generate(idx=test_sample, key=jr.key(42))
        text = decode([int(x) for x in out])
        evals_table.append([text])
    wandb.log({"text": wandb.Table(["text"], data=evals_table), "eval_loss": eval_loss})
    return eval_loss


def main():
    model = GPT.make(jr.key(0), model_config)
    model_params = eqx.filter(model, eqx.is_array)

    from opts import Adam, HeunGrad

    opt_state = HeunGrad(Adam())

    n_model_params = jax.tree.map(lambda x: x.size, model_params)
    n_model_params = sum(jax.tree.leaves(n_model_params))

    print(f"Model has {n_model_params/1_000_000:.2f}M parameters")
    eval_loss = float(jnp.nan)

    devices = mesh_utils.create_device_mesh((1, jax.device_count(), 1))

    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    model, opt_state = eqx.filter_shard((model, opt_state), replicated)

    train_generator = np.random.default_rng(42)
    eval_generator = np.random.default_rng(69)

    def checkpoint(i):
        Path("checkpoints").mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(f"checkpoints/model_{i}.eqx", model)
        wandb.log_artifact("checkpoint", f"checkpoints/model_{i}.eqx")

    def load_train_batches():
        return get_batches(
            "train",
            train_generator,
            shape=(
                train_config.n_grad_accumulation,
                train_config.batch_size,
            ),
        )

    X, y = load_train_batches()
    for i in (pbar := tqdm(range(train_config.train_for))):
        data_key, fwd_key = jr.split(jr.key(i))

        t = time.time()

        X, y = eqx.filter_shard((jnp.array(X), jnp.array(y)), sharding)
        updates, opt_state, loss = opt_state.step_with(step_fn, fwd_key)
        model = eqx.apply_updates(model, updates)
        X, y = load_train_batches()
        loss = float(loss.mean())

        pbar.set_description(
            f"loss:{loss:.2f} / eval:{eval_loss:.2f} | step:{(time.time() - t)*1e3:.2f}ms"
        )
        wandb.log({"loss": loss})

        chckp_freq = train_config.train_for // run_config.times_to_checkpoint
        if i % chckp_freq == chckp_freq - 1:
            checkpoint(i)

        eval_freq = train_config.train_for // run_config.times_to_eval
        if i % eval_freq == eval_freq - 1:
            eval_loss = eval_fn(
                eqx.nn.inference_mode(model), eval_generator, train_config.batch_size, sharding
            )

    checkpoint(train_config.train_for)


if __name__ == "__main__":
    main()
