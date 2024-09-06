import equinox as eqx
import jax


class GPTConfig(eqx.Module):
    context_len: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 368
    dropout: float = 0.2
    conditional_limit: float = 0.5

    @classmethod
    def from_preset(cls, name: str):
        match name:
            case "chargpt":
                return GPTConfig()
            case "gpt2":
                return GPTConfig(
                    context_len=1024,
                    vocab_size=50384,
                    n_embed=768,
                    n_layer=12,
                    n_head=12,
                )
            case _:
                raise AssertionError("Only 'chargpt' or 'gpt2' are allowed as presets!")


class TrainConfig(eqx.Module):
    batch_size: int = 16
    lr_config: dict = eqx.field(
        default_factory=lambda: {
            "init_value": 1e-3,
            "peak_value": 1e-3,
            "warmup_steps": 100,
            "decay_steps": 5000,
            "end_value": 1e-4,
        }
    )
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
                    batch_size=512,  # from gpt2 paper
                    train_for=600_000,
                    lr_config=eqx.field(
                        default_factory=lambda: {
                            "init_value": 6e-4,
                            "peak_value": 6e-4,
                            "warmup_steps": 1000,
                            "decay_steps": 600_000,
                            "end_value": 6e-5,
                        }
                    ),
                    dataset_name="openwebtext",
                )
            case _:
                raise AssertionError("Only 'chargpt' or 'gpt2' are allowed as presets!")


class RunConfig(eqx.Module):
    n_devices: int = 1  # number of devices available
    n_updates_on_device: int = 4  # how many steps to do without moving data to CPU
    times_to_checkpoint: int = 2  # how many times to checkpoint throughout the training
    times_to_eval: int = 20  # how many times to eval throughout training
    n_batches_in_eval: int = 20  # how many batches in eval

    def __post_init__(self):
        self.n_devices = jax.device_count()

    @classmethod
    def from_preset(cls, name: str):
        match name:
            case "chargpt":
                return RunConfig()
            case "gpt2":
                return RunConfig(times_to_eval=100, times_to_checkpoint=50)
            case _:
                raise AssertionError("Only 'chargpt' or 'gpt2' are allowed as presets!")
