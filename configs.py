import equinox as eqx
import jax
import jax.numpy as jnp


class GPTConfig(eqx.Module):
    context_len: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_heads: int = 6
    n_embed: int = 384
    dropout: float = 0.2
    dtype: jnp.dtype = jnp.bfloat16

    @classmethod
    def from_preset(cls, name: str):
        match name:
            case "chargpt":
                return GPTConfig()
            case "gpt2":
                return GPTConfig(
                    context_len=1024,
                    vocab_size=50257,
                    n_embed=768,
                    n_layers=12,
                    n_heads=12,
                    dropout=0.0,
                )
            case _:
                raise AssertionError("Only 'chargpt' or 'gpt2' are allowed as presets!")


class TrainConfig(eqx.Module):
    batch_size: int = 64
    n_grad_accumulation: int = 1  # for how many steps to accumulate gradients
    lr_config: dict = eqx.field(
        default_factory=lambda: {
            "init_value": 1e-3,
            "peak_value": 1e-3,
            "warmup_steps": 100,
            "decay_steps": 5000,
            "end_value": 1e-4,
        }
    )
    weight_decay: float = 1e-2
    global_norm: float = 1.0
    train_for: int = 5000
    dataset_name: str = "shakespear-char"

    @classmethod
    def from_preset(cls, name: str):
        match name:
            case "chargpt":
                return TrainConfig()
            case "gpt2":
                return TrainConfig(
                    batch_size=256,  # gpt2 paper - 512
                    n_grad_accumulation=1,
                    train_for=600_000,
                    lr_config={
                        "init_value": 2e-3,
                        "peak_value": 2e-3,
                        "warmup_steps": 1000,
                        "decay_steps": 600_000,
                        "end_value": 5e-5,
                    },
                    dataset_name="openwebtext",
                )
            case _:
                raise AssertionError("Only 'chargpt' or 'gpt2' are allowed as presets!")


class RunConfig(eqx.Module):
    n_devices: int = 1  # number of devices available
    times_to_checkpoint: int = 2  # how many times to checkpoint throughout the training
    times_to_eval: int = 10  # how many times to eval throughout training
    n_batches_in_eval: int = 20  # how many batches in eval

    def __post_init__(self):
        self.n_devices = jax.device_count()

    @classmethod
    def from_preset(cls, name: str):
        match name:
            case "chargpt":
                return RunConfig()
            case "gpt2":
                return RunConfig(times_to_eval=200, times_to_checkpoint=10, n_batches_in_eval=100)
            case _:
                raise AssertionError("Only 'chargpt' or 'gpt2' are allowed as presets!")
