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
from helpers import WandbLogger, auto_batch_size_wrapper
from model import GPT

initialise_tracking()

wandb = WandbLogger(use_wandb=False, name="nano_jax_gpt_test")

model_config: GPTConfig = GPTConfig.from_preset("chargpt")
train_config: TrainConfig = TrainConfig.from_preset("chargpt")
run_config: RunConfig = RunConfig.from_preset("chargpt")

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
    (model_d, opt_state, _), loss = eqxi.scan(
        eqx.Partial(scan_fn, model_s=model_s), (model_d, opt_state, key), (X, y), kind="lax"
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


def main(batch_size=train_config.batch_size, *, exit_after_first_step=False):
    model = GPT(jr.key(0), model_config)
    n_model_params = jax.tree.map(lambda x: x.size, eqx.filter(model, eqx.is_array))
    n_model_params = sum(jax.tree_leaves(n_model_params))
    print(f"Model has {n_model_params/1_000_000:.2f}M parameters")
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    evals_table = []
    eval_loss = jnp.nan

    devices = mesh_utils.create_device_mesh((1, jax.device_count(), 1))  # reps x batch x data
    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    model = eqx.filter_shard(model, replicated)

    def checkpoint(i):
        Path("checkpoints").mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(f"checkpoints/model_{i}.eqx", model)

    n_steps = train_config.train_for // run_config.n_updates_on_device
    for i in (pbar := tqdm(range(n_steps))):
        data_key, fwd_key = jr.split(jr.key(i))
        t = time.time()
        X, y = get_batches(
            "train",
            data_key,
            shape=(
                run_config.n_updates_on_device,
                batch_size,
            ),
        )
        loading_data_time = time.time() - t

        # step
        t = time.time()
        X, y = eqx.filter_shard((jnp.array(X), jnp.array(y)), sharding)
        model, opt_state, loss = eqx.filter_jit(upd_fn, donate="all")(
            model, opt_state, X, y, fwd_key, sharding
        )
        loss.block_until_ready()
        del X, y, fwd_key  # since donate="all" make sure to GC the vars too

        step_time = (time.time() - t) / run_config.n_updates_on_device

        # log
        pbar.set_description(
            f"loss:{loss.mean():.2f} / eval:{eval_loss:.2f} |"
            f"loading:{loading_data_time:.2f}s / step:{step_time:.2f}s"
        )
        if exit_after_first_step:
            return
        wandb.log({"loss": loss.mean(), "step": i})

        # if we want to checkpoint..
        if i % (n_steps // run_config.times_to_checkpoint) == 1:
            checkpoint(i)

        # if we want to eval...
        if i % (n_steps // run_config.times_to_eval) == 1:
            X, y = get_batches("test", jr.key(32), shape=(run_config.n_batches_in_eval, batch_size))
            X, y = eqx.filter_shard((jnp.array(X), jnp.array(y)), sharding)
            eval_loss = eqx.filter_jit(evaluate)(eqx.nn.inference_mode(model), X, y, sharding)

            test_sample = get_batches("test", key=jr.key(1), shape=(1,))[0][0]
            out = eqx.filter_jit(model.generate)(idx=test_sample, key=jr.key(42))
            text = decode([int(x) for x in out])
            evals_table.append([i, text])
            wandb.log(
                {"text": wandb.Table(["step", "text"], data=evals_table), "eval_loss": eval_loss},
                commit=True,
            )

    checkpoint(train_config.train_for // run_config.n_updates_on_device)


if __name__ == "__main__":
    optim = optax.chain(
        optax.clip_by_global_norm(train_config.global_norm),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(**train_config.lr_config), weight_decay=1e-1
        ),
    )
    auto_batch_size_wrapper(
        main, batch_size=train_config.batch_size, n_devices=run_config.n_devices
    )
