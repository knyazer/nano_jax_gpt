import gc
import time

import equinox as eqx
import jax

import wandb


def auto_batch_size_wrapper(fn, batch_size=0, n_devices=1):
    if batch_size != 0:
        fn(batch_size)
        return

    batch_size = n_devices * 4
    print(f"""\033[34m
Starting with a batch size of {batch_size}.
Running a single step of training...
\033[0m""")
    for _ in range(10):
        try:
            # run a single step of training
            fn(batch_size, exit_after_first_step=True)
            eqx.clear_caches()
            gc.collect()
            time.sleep(0.5)  # wait for the gc...

            # load memory stats
            memory_stats = jax.local_devices()[0].memory_stats()
            assert memory_stats is not None, "Your devices don't support auto batch sizing :("
            available_memory = float(memory_stats["bytes_limit"])
            peak_memory = float(
                max(memory_stats["peak_bytes_in_use"], memory_stats["peak_bytes_reserved"])
            )
            print(memory_stats)

            if peak_memory >= available_memory * 0.90:
                break
            # we try to avoid rematerialization (perf reasons), so no overshooting allowed
            new_batch_size = int(batch_size * (available_memory * 0.95 / peak_memory))
            new_batch_size = (new_batch_size // n_devices) * n_devices  # round to num of devices
            if new_batch_size <= batch_size:
                break
            batch_size = new_batch_size
            print(f"\033[34m\nNew batch size: {batch_size}\n\033[0m")

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"""
Failed due to an OOM: jax cannot resurrect after OOM usually, hence,
please, restart with a specified batch size lower than {batch_size}.""")
                return
            raise e  # noqa

    print(f"\n\033[32mFinal batch size: {batch_size}. Starting the training...\033[0m\n")

    fn(batch_size)


class WandbLogger:
    def __init__(self, *, use_wandb=True, **kws):
        self.use_wandb = use_wandb
        self._kws = kws

    def Table(self, *args, **kwargs):  # noqa
        return wandb.Table(*args, **kwargs)

    def log_artifact(self, name, path):
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(local_path=path, name=f"model:{path}")
        artifact.save()

    def log(self, *args, **kwargs):
        if self.use_wandb:
            if wandb.run is None:
                wandb.init(
                    project="nano_jax_gpt",
                    settings=wandb.Settings(code_dir="."),
                    **self._kws,
                )

            wandb.log(*args, **kwargs)
