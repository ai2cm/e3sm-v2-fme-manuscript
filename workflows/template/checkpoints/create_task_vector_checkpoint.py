import argparse
import logging
import os

import dacite
import fme
import torch
import yaml
from fme.core.data_loading.getters import get_data_loader
from fme.fcn_training.train_config import TrainConfig


def main(yaml_config, finetuned_weights, lambda_param):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)

    config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(
        os.path.join(config.experiment_dir, "task-vector-checkpoint-config.yaml"), "w"
    ) as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    config.configure_logging(log_filename="create_task_vector_checkpoint.log")

    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    pre_ckpt = torch.load(
        data["stepper"]["parameter_init"]["weights_path"],
        map_location=fme.get_device(),
    )
    pre_state_dict = pre_ckpt["stepper"]["module"]

    ft_ckpt = torch.load(
        finetuned_weights,
        map_location=fme.get_device(),
    )
    ft_state_dict = ft_ckpt["stepper"]["module"]

    # compute pt_state + lambda * (ft_state - pt_state)
    new_state = {"module": {}}
    for key in ft_state_dict.keys():
        if key not in pre_state_dict:
            logging.warning(
                f"Finetuned model parameter {key} not found in pretrained model state_dict"
            )
            new_state[key] = ft_state_dict[key]
        else:
            new_state["module"][key] = pre_state_dict[key] + lambda_param * (
                ft_state_dict[key] - pre_state_dict[key]
            )

    logging.info("Creating a new ACE checkpoint from finetuning task vector.")
    logging.info(
        f"Pretrained weights path: {config.stepper.parameter_init.weights_path}"
    )
    logging.info(f"Finetuned weights path: {finetuned_weights}")
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

    # stepper will be initialized according to yaml_config
    stepper = config.stepper.get_stepper(
        img_shape=img_shape,
        area=valid_data.area_weights,
        sigma_coordinates=valid_data.sigma_coordinates,
    )
    # replace stepper's state with: pt_state + lambda * (ft_state - pt_state)
    stepper.load_state(new_state)

    ckpt_path = os.path.join(
        config.checkpoint_dir, f"task_vector_ckpt_lambda{lambda_param}.tar"
    )

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
    parser.add_argument("--finetuned_weights", required=True, type=str)
    parser.add_argument("--lambda_param", default=0.5, type=float)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
        finetuned_weights=args.finetuned_weights,
        lambda_param=args.lambda_param,
    )
