import os
from itertools import product

import click
import yaml


@click.command()
@click.option("--batch-size", "-b", type=int, multiple=True)
@click.option("--max-epochs", "-m", type=int, multiple=True)
@click.option("--lr", "-l", type=float, multiple=True)
@click.option("--config-dir", "-c", type=str)
@click.option("--base-config", type=str)
@click.option("--debug", is_flag=True)
def main(batch_size, max_epochs, lr, config_dir, base_config, debug):
    hparams = product(batch_size, max_epochs, lr)
    with open(base_config, "r") as f:
        data = yaml.safe_load(f)
    for bs, epochs, lr in hparams:
        group_dir = f"batch_size{bs}-max_epochs{epochs}"
        name_dir = f"lr{lr:.2e}"
        data["max_epochs"] = epochs
        data["train_data"]["batch_size"] = bs
        data["validation_data"]["batch_size"] = bs
        data["stepper"]["optimization"]["lr"] = lr
        subdir = os.path.join(config_dir, group_dir, name_dir)
        if debug:
            print(subdir, "-" * len(subdir), data, "", sep="\n")
        else:
            os.makedirs(subdir)
            with open(os.path.join(subdir, "train-config.yaml"), "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
