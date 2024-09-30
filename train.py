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

jax.config.update("jax_threefry_partitionable", True)  # noqa

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
parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

wandb = WandbLogger(use_wandb=(jax.process_index() == 0), name=f"testing-{args.model}")

model_config: GPTConfig = GPTConfig.from_preset(args.model)
train_config: TrainConfig = TrainConfig.from_preset(args.model)
run_config: RunConfig = RunConfig.from_preset(args.model)


def _jax_log(data, cond):
    with jax.ensure_compile_time_eval():
        if cond:
            wandb.log(data, commit=False)


jax_log = eqx.Partial(lambda *args: jax.debug.callback(_jax_log, *args))

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

    grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads, is_leaf=eqx.is_inexact_array_like)

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss.astype(jnp.float32)


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
    wandb.log(
        {"text": wandb.Table(["text"], data=evals_table), "eval_loss": eval_loss}, commit=False
    )
    return eval_loss


def main():  # noqa
    model = GPT.make(jr.key(0), model_config)
    if args.resume:
        model = eqx.tree_deserialise_leaves("checkpoint.eqx", model)
        starting_index = 24_000
    else:
        starting_index = 0
    model_params = eqx.filter(model, eqx.is_array)

    class AdamWState(eqx.Module):
        m: Any
        v: Any
        t: Any
        prev_grads: Any
        prev_upd: Any
        fast_v: Any

    class AdamW(eqx.Module):
        lr_config: dict = eqx.field(static=True)
        weight_decay: float = eqx.field(static=True)
        global_norm: float = eqx.field(static=True)
        beta1: float = eqx.field(static=True)
        beta2: float = eqx.field(static=True)
        epsilon: float = eqx.field(static=True)
        start_t: int = eqx.field(static=True)

        def __init__(self, lr_config, weight_decay, global_norm, t=0):
            self.lr_config = lr_config
            self.weight_decay = weight_decay
            self.global_norm = global_norm
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.start_t = t

        def init(self, params, t=0):
            return AdamWState(
                m=jax.tree.map(jnp.zeros_like, params),
                v=jax.tree.map(jnp.zeros_like, params),
                t=jnp.array(t, dtype=jnp.int32),
                prev_grads=jax.tree.map(jnp.zeros_like, params),
                prev_upd=jax.tree.map(jnp.zeros_like, params),
                fast_v=jax.tree.map(jnp.zeros_like, params),
            )

        def warmup_cosine_decay(self, t):
            warmup_steps = self.lr_config["warmup_steps"]
            decay_steps = self.lr_config["decay_steps"]
            init_value = self.lr_config["init_value"]
            peak_value = self.lr_config["peak_value"]
            end_value = self.lr_config["end_value"]

            def warmup(t):
                return init_value + (peak_value - init_value) * ((t - self.start_t) / warmup_steps)

            def decay(t):
                t = t - warmup_steps
                total_decay_steps = decay_steps - warmup_steps
                cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * t / total_decay_steps))
                return end_value + (peak_value - end_value) * cosine_decay

            lr = jax.lax.cond(t < warmup_steps + self.start_t, warmup, decay, t)
            return lr

        def update(self, grads, state, params):
            def l2(x):
                return jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda g: jnp.sum(g**2), x))))

            def corr(a, b):
                prod = jax.tree_map(lambda x, y: x * y, a, b)
                return sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.sum(x), prod))) / (
                    l2(a) * l2(b)
                )

            def clip(x, norm):
                global_l2_norm = l2(x)
                return jax.tree.map(
                    lambda g: jax.lax.cond(
                        global_l2_norm > norm,
                        lambda: g * norm / (global_l2_norm + 1e-6),
                        lambda: g,
                    ),
                    x,
                )

            # grads also differs; grads if its an intermediate step, grads is just grads
            # otherwise it is -prev_grads * 0.5 + grads
            # and update is just -old_update + new_update

            t = state.t + 1
            lr = self.warmup_cosine_decay(t)

            unscaled_grads = grads
            grads = jax.lax.cond(
                jnp.mod(t, 2) == 0,
                lambda: jax.tree.map(
                    lambda g, pg: g * 0.5 + pg * 0.5,
                    grads,
                    state.prev_grads,
                ),
                lambda: jax.tree.map(lambda g: g * 9.0, grads),
            )
            grads = clip(grads, self.global_norm)

            jax_log(
                {"raw_grad_norm_s1": l2(grads), "grad_scaled_norm_s1": l2(unscaled_grads)},
                jnp.mod(t, 2) == 1,
            )

            jax_log(
                {"raw_grad_norm_s2": l2(grads), "grad_scaled_norm_s2": l2(unscaled_grads)},
                jnp.mod(t, 2) == 0,
            )

            def update_moment(m, g):
                return self.beta1 * m + (1.0 - self.beta1) * g

            def update_velocity(v, g):
                return self.beta2 * v + (1.0 - self.beta2) * (g**2)

            def compute_update(m, v, p):
                m_hat = m / (1.0 - self.beta1 ** (t - self.start_t))
                v_hat = v / (1.0 - self.beta2 ** (t - self.start_t))
                # we assume conditioning does not change much - or fixed
                update = -lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
                if eqx.is_inexact_array(p) and p.ndim >= 2:
                    update -= lr * self.weight_decay * p
                return update

            new_m = jax.tree.map(update_moment, state.m, grads)

            def application_update():
                new_v = jax.tree.map(update_velocity, state.v, grads)
                updates = jax.tree.map(compute_update, new_m, new_v, params)
                # updates are just the new updates - old_updates
                mod_updates = jax.tree.map(lambda x, y: x - y, updates, state.prev_upd)

                err = l2(jax.tree.map(lambda g, pg: g - pg, grads, state.prev_grads))
                err_rel = corr(grads, state.prev_grads)  # 1 is linear fn, 0 is noise, -1 opposite
                jax_log(
                    {
                        "solver_error": err,
                        "solver_error_correlation": err_rel,
                        "random_p_grad_1": state.prev_grads.blocks[3].proj_fc.weight.ravel()[157],
                        "random_p_grad_2": unscaled_grads.blocks[3].proj_fc.weight.ravel()[157],
                        "G^2": l2(new_v),
                        "M": l2(new_m),
                    },
                    jnp.mod(t, 2) == 0,
                )

                return mod_updates, AdamWState(
                    m=new_m,
                    v=new_v,
                    t=t,
                    prev_grads=unscaled_grads,
                    prev_upd=updates,
                    fast_v=state.fast_v,
                )

            def compute_intermediate_update(g, v, p):
                v_hat = v / (1.0 - self.beta2 ** (t - self.start_t))
                update = -lr * g / (jnp.sqrt(v_hat) + self.epsilon)
                if eqx.is_inexact_array(p) and p.ndim >= 2:
                    update -= lr * self.weight_decay * p
                return update

            def heuns_update():
                # heuns update, we don't update any opt state here, only the update store
                new_v = jax.tree.map(update_velocity, state.fast_v, unscaled_grads)
                updates = jax.tree.map(compute_intermediate_update, new_m, new_v, params)
                return updates, AdamWState(
                    m=state.m,
                    v=state.v,
                    t=t,
                    prev_grads=unscaled_grads,
                    prev_upd=updates,
                    fast_v=new_v,
                )

            return jax.lax.cond(jnp.mod(t, 2) == 0, application_update, heuns_update)

    optim = AdamW(
        lr_config=train_config.lr_config,
        weight_decay=train_config.weight_decay,
        global_norm=train_config.global_norm,
        t=starting_index,
    )

    n_model_params = jax.tree.map(lambda x: x.size, model_params)
    n_model_params = sum(jax.tree.leaves(n_model_params))

    print(f"Model has {n_model_params/1_000_000:.2f}M parameters")
    opt_state = optim.init(model_params, t=starting_index)
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

    def skip_train_batches(n):
        for _ in range(n):
            train_generator.integers(
                low=0,
                high=1000,
                size=(
                    int(
                        np.prod(
                            (
                                train_config.n_grad_accumulation,
                                train_config.batch_size,
                            )
                        )
                    ),
                ),
            )

    skip_train_batches(starting_index)
    X, y = load_train_batches()
    for i in (pbar := tqdm(range(starting_index, train_config.train_for))):
        data_key, fwd_key = jr.split(jr.key(i))

        t = time.time()

        X, y = eqx.filter_shard((jnp.array(X), jnp.array(y)), sharding)
        model, opt_state, loss = step_fn(model, optim, opt_state, X, y, fwd_key)
        X, y = load_train_batches()
        loss_var = jnp.log(loss.std() + 1e-13)
        loss = float(loss.mean())

        pbar.set_description(
            f"loss:{loss:.2f} / eval:{eval_loss:.2f} | step:{(time.time() - t)*1e3:.2f}ms"
        )
        wandb.log({"step": i, "loss": loss, "loss_var": loss_var}, commit=False)

        # since our method is multi-step, we are interested only in the even steps
        if (i - starting_index) % 2 == 0:
            wandb.log({"clean_loss": loss, "clean_var": loss_var}, commit=False)

        chckp_freq = train_config.train_for // run_config.times_to_checkpoint
        if i % chckp_freq == chckp_freq - 1:
            checkpoint(i)

        eval_freq = train_config.train_for // run_config.times_to_eval
        if i % eval_freq == eval_freq - 1:
            eval_loss = eval_fn(
                eqx.nn.inference_mode(model), eval_generator, train_config.batch_size, sharding
            )
        wandb.log({}, commit=True)

    checkpoint(train_config.train_for)


if __name__ == "__main__":
    with jax.log_compiles():
        main()
