Python scripts for running HPO sweeps for the following hyperparameters:

- `-b`, `--batch-size`: the batch size.
- `-m`, `--max-epochs`: max number of epochs the LR scheduler, i.e. `T_max` for `CosineAnnealingLR`.
- `l`, `--lr`: the learning rate.
- `-c`, `--config-dir`: top-level config directory (actual configs will be placed in sub-directories).

Python package requirements:

- `click`
- `yaml`

## `make_configs.py` example usage

Create a template `train-config.yaml` for `--base-config`, and specify HPO grid:

``` shell
python -u make_configs.py \
    -b 8 -b 16 \
    -m 50 -m 75 \
    -l 2e-4 -l 3e-4 -l 4e-4 \
    --config-dir $SCRATCH/ai2/config/topo-hpo \
    --base-config $SCRATCH/ai2/config/train/train-config.yaml
```

`--config-dir` is the directory where the HPO configs will be saved in
sub-directories determined by combinations of `-b` and `-m` (first level) and
`-l` (second level); e.g., in the above example one of the configs will be
created at `batch_size8-max_epochs50/lr2.00e-4/train-config.yaml`.

## `submit_configs.py` usage

The HPO CLI args to `submit_configs.py` mirror those to `make_configs.py`, i.e.
when using the same values for `--config-dir`, `-b`, `-m`, and `-l`, the script
will look for `train-config.yaml` files in the location where `make_configs.py`
saved them. Each config will be submitted as a separate job with a unique job
name.

``` shell
python -u submit_configs.py \
    -b 8 -b 16 \
    -m 50 -m 75 \
    -l 2e-4 -l 3e-4 -l 4e-4 \
    -c $SCRATCH/ai2/config/topo-hpo \
    -i fme:b5c4b6e \
    -d $SCRATCH/ai2/datasets/e3smv2-1deg-gaussian-fme-40yr-topo \
    -t 24:00:00 \
    --group-prefix topo \
    --randomize
```

Additional args are:

- `--sbatch-script`, `-s`: Optional path to an sbatch script. Default is `../sbatch-scripts/submit.sh`.
- `--image`, `-i`: The `podman-hpc` image to use.
- `--data-dir`, `-d`: The data directory.
- `--time`, `-t`: time limit in `HH:MM:SS` format (default `06:00:00`).
- `--group-prefix`: an optional prefix to add to `WANDB_RUN_GROUP` for uniqueness.
- `--randomize`: flag which randomizes the order of job submissions.
- `--debug`: flag which, when provided, prints the `sbatch` command that would
  run, without actually submitting the job.
