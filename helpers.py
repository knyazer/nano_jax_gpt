import wandb


class WandbLogger:
    def __init__(self, *, use_wandb=True, **kws):
        self.use_wandb = use_wandb
        self._kws = kws

    def Table(self, *args, **kwargs):  # noqa
        return wandb.Table(*args, **kwargs)

    def log_artifact(self, name, path):
        if self.use_wandb:
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
