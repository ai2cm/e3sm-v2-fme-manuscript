import os
import subprocess
from itertools import product
from random import shuffle
from time import sleep

import click


@click.command()
@click.option("--batch-size", "-b", type=int, multiple=True)
@click.option("--max-epochs", "-m", type=int, multiple=True)
@click.option("--lr", "-l", type=float, multiple=True)
@click.option("--config-dir", "-c", type=str)
@click.option("--image", "-i", type=str)
@click.option("--data-dir", "-d", type=str)
@click.option("--time", "-t", type=str, default="06:00:00")
@click.option("--group-prefix", type=str, default=None)
@click.option("--randomize", is_flag=True)
@click.option("--debug", is_flag=True)
def main(
    batch_size,
    max_epochs,
    lr,
    config_dir,
    image,
    data_dir,
    time,
    group_prefix,
    randomize,
    debug,
):
    hparams = list(product(batch_size, max_epochs, lr))
    if randomize:
        shuffle(hparams)
    for bs, epochs, lr in hparams:
        group_dir = f"batch_size{bs}-max_epochs{epochs}"
        name_dir = f"lr{lr:.2e}"
        subdir = os.path.join(config_dir, group_dir, name_dir)
        assert os.path.exists(subdir), f"{subdir} doesn't exist!"
        if group_prefix is not None:
            group_name = f"{group_prefix}-{group_dir}"
        else:
            group_name = group_prefix
        cmd = (
            f"sbatch -J {group_name}-{name_dir} --dependency=singleton -N 1 -t {time} "
            f"submit.sh -c {subdir} -i {image} -g {group_name} -n {name_dir} -d {data_dir} "
            "-t train"
        )
        print(cmd)
        if not debug:
            output = subprocess.check_output(cmd, shell=True)
            print(output.decode("utf-8"))
            # sleep for a couple seconds to avoid overwhelming Slurm
            sleep(2)


if __name__ == "__main__":
    main()
