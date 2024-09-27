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

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

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

    class AdamWState(eqx.Module):
        m: Any
        v: Any
        t: Any
        prev_grads: Any
        prev_upd: Any

    class AdamW(eqx.Module):
        lr_config: dict = eqx.field(static=True)
        weight_decay: float = eqx.field(static=True)
        global_norm: float = eqx.field(static=True)
        beta1: float = eqx.field(static=True)
        beta2: float = eqx.field(static=True)
        epsilon: float = eqx.field(static=True)

        def __init__(self, lr_config, weight_decay, global_norm):
            self.lr_config = lr_config
            self.weight_decay = weight_decay
            self.global_norm = global_norm
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

        def init(self, params):
            return AdamWState(
                m=jax.tree.map(jnp.zeros_like, params),
                v=jax.tree.map(jnp.zeros_like, params),
                t=jnp.array(0, dtype=jnp.int32),
                prev_grads=jax.tree.map(jnp.zeros_like, params),
                prev_upd=jax.tree.map(jnp.zeros_like, params),
            )

        def warmup_cosine_decay(self, t):
            warmup_steps = self.lr_config["warmup_steps"]
            decay_steps = self.lr_config["decay_steps"]
            init_value = self.lr_config["init_value"]
            peak_value = self.lr_config["peak_value"]
            end_value = self.lr_config["end_value"]

            def warmup(t):
                return init_value + (peak_value - init_value) * (t / warmup_steps)

            def decay(t):
                t = t - warmup_steps
                total_decay_steps = decay_steps - warmup_steps
                cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * t / total_decay_steps))
                return end_value + (peak_value - end_value) * cosine_decay

            lr = jax.lax.cond(t < warmup_steps, warmup, decay, t)
            return lr

        def update(self, grads, state, params):
            global_l2_norm = jnp.sqrt(
                sum(jax.tree.leaves(jax.tree.map(lambda g: jnp.sum(g**2), grads)))
            )
            grads = jax.lax.cond(
                global_l2_norm > self.global_norm,
                lambda: jax.tree.map(
                    lambda g: g * (self.global_norm / (global_l2_norm + 1e-6)), grads
                ),
                lambda: grads,
            )

            t = state.t + 1
            lr = self.warmup_cosine_decay(t)

            def update_moment(m, g):
                return self.beta1 * m + (1.0 - self.beta1) * g

            def update_velocity(v, g):
                return self.beta2 * v + (1.0 - self.beta2) * (g**2)

            def compute_update(m, v, p):
                m_hat = m / (1.0 - self.beta1**t)
                v_hat = v / (1.0 - self.beta2**t)
                update = -lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
                if eqx.is_inexact_array(p) and p.ndim >= 2:
                    update -= lr * self.weight_decay * p
                return update

            # steps!

            def step1():
                # just do a single step with grads
                new_m = jax.tree.map(update_moment, state.m, grads)
                new_v = jax.tree.map(update_velocity, state.v, grads)
                updates = jax.tree.map(compute_update, new_m, new_v, params)

                return updates, AdamWState(state.m, state.v, t, grads, updates)  # pass the grads

            def step2():
                # we compute the average
                avg_grads = jax.tree.map(lambda g, pg: g * 0.5 + pg * 0.5, grads, state.prev_grads)

                # do a step with the new estimate
                new_m = jax.tree.map(update_moment, state.m, avg_grads)
                new_v = jax.tree.map(update_velocity, state.v, avg_grads)
                updates = jax.tree.map(compute_update, new_m, new_v, params)

                # we want to apply the update to the original params (step 1), so sub the old update
                updates = jax.tree.map(lambda u, pu: u - pu, updates, state.prev_upd)
                return updates, AdamWState(state.m, state.v, t, grads, updates)  # still frozen

            def step3():
                # another (and last) implicit step
                avg_grads = jax.tree.map(lambda g, pg: g * 0.5 + pg * 0.5, grads, state.prev_grads)

                new_m = jax.tree.map(update_moment, state.m, avg_grads)
                new_v = jax.tree.map(update_velocity, state.v, avg_grads)
                updates = jax.tree.map(compute_update, new_m, new_v, params)

                updates = jax.tree.map(lambda u, pu: u - pu, updates, state.prev_upd)
                return updates, AdamWState(new_m, new_v, t, grads, updates)  # unfreeze

            return jax.lax.switch(jnp.mod(t, 5), [step1, step2, step2, step2, step3])

    optim = AdamW(
        lr_config=train_config.lr_config,
        weight_decay=train_config.weight_decay,
        global_norm=train_config.global_norm,
    )

    n_model_params = jax.tree.map(lambda x: x.size, model_params)
    n_model_params = sum(jax.tree.leaves(n_model_params))

    print(f"Model has {n_model_params/1_000_000:.2f}M parameters")
    opt_state = optim.init(model_params)
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
        model, opt_state, loss = step_fn(model, optim, opt_state, X, y, fwd_key)
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
