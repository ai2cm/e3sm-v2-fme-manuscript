import argparse
import logging
import os

import dacite
import fme
import torch
import yaml
from fme.core.data_loading.getters import get_data_loader
from fme.fcn_training.train_config import TrainConfig


def main(yaml_config):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)

    config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(os.path.join(config.experiment_dir, "checkpoint-config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    config.configure_logging(log_filename="create_checkpoint.log")

    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    logging.info("Creating a new ACE checkpoint from pretrained weights.")
    logging.info(f"Weights path: {config.stepper.parameter_init.weights_path}")
    logging.info(f"Validation data path: {config.validation_loader.dataset.data_path}")
    logging.info(f"Global means path: {config.stepper.normalization.global_means_path}")
    logging.info(f"Global STDs path: {config.stepper.normalization.global_stds_path}")

    data_requirements = config.stepper.get_data_requirements(
        n_forward_steps=config.n_forward_steps
    )

    valid_data = get_data_loader(
        config.validation_loader,
        requirements=data_requirements,
        train=False,
    )

    for batch in valid_data.loader:
        shapes = {k: v.shape for k, v in batch.data.items()}
        for value in shapes.values():
            img_shape = value[-2:]
            break
        break

    # stepper will be initialized from the pretrained weights and the specified
    # normalization
    stepper = config.stepper.get_stepper(
        img_shape=img_shape,
        area=valid_data.area_weights,
        sigma_coordinates=valid_data.sigma_coordinates,
    )

    # sanity check stepper's weights
    ckpt = torch.load(
        data["stepper"]["parameter_init"]["weights_path"],
        map_location=fme.get_device(),
    )

    state_dict = stepper.module.state_dict()
    for key, tensor in state_dict.items():
        assert torch.all(
            tensor.detach() == ckpt["stepper"]["module"][key]
        ), f"Parameter {key} doesn't match"

    for key in ckpt["stepper"]["module"].keys():
        if key not in state_dict:
            logging.warning(
                f"Pretrained model parameter {key} not found in stepper state_dict"
            )

    ckpt_path = os.path.join(config.checkpoint_dir, "pretrained_ckpt.tar")

    logging.info(f"Saving newly created checkpoint to {ckpt_path}")
    torch.save(
        {
            "num_batches_seen": 0,
            "epoch": 0,
            "stepper": stepper.get_state(),
        },
        ckpt_path,
    )
    logging.info("Checkpoint created... DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True, type=str)

    args = parser.parse_args()

    main(yaml_config=args.yaml_config)
