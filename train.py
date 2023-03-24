# --------------------------------------------
# Train pi-GAN. Supports distributed training.
# --------------------------------------------
import argparse
from dis import disco
import os
import socket
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import datasets
import discriminators
from utils import Trainer
import pickle
import random

def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def setup(rank, world_size, free_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_dataloader(metadata: dict, ddp: bool, world_size: int, rank: int):
    if ddp:
        dataloader, CHANNELS = datasets.get_dataset_distributed(
            metadata, world_size, rank
        )
    else:
        dataloader, CHANNELS = datasets.get_dataset(metadata)
    return dataloader


# @profile
def train(rank, opt, world_size=1, free_port=0):
    # cnt = 0
    if opt.ddp:
        setup(rank, world_size, free_port)

    trainer = Trainer(opt, rank, world_size)
    ### load models
    trainer.load_models()

    trainer.write_train_settings()
    generator = trainer.generator
    if trainer.metadata["enable_discriminator"]:
        discriminator = trainer.discriminator
    torch.manual_seed(rank)
    random.seed(rank)
    trainer.update_metadata()
    dataloader = get_dataloader(trainer.metadata, opt.ddp, world_size, rank)
    steps_elasped_time = 0
    # path_ls = []
    for _ in range(opt.n_epochs):
        epoch_start_time = time.time()
        if rank == 0:
            print(f"---- {generator.epoch}_th epoch ----")
        # Set learning rates
        trainer.set_learning_rate()
        for sample in dataloader:
            metadata_updated = trainer.update_metadata()
            if metadata_updated:
                print(f"Metadata updated at step {generator.step}. Reload dataset.")
                dataloader = get_dataloader(trainer.metadata, opt.ddp, world_size, rank)
                break
            # Note: to avoid scaler._scale is None bug
            if trainer.scaler.get_scale() < 1:
                trainer.scaler.update(1.0)

            trainer.generator_ddp.train()
            if trainer.metadata["enable_discriminator"]:
                trainer.discriminator_ddp.train()

            # alpha as in fade-in step of pregressiveGAN
            trainer.set_alpha()

            step_start_time = time.time()
            if trainer.metadata["enable_discriminator"]:
                trainer.train_discriminator(sample)
            ### train generator
            # with torch.autograd.set_detect_anomaly(True):
            trainer.train_generator(sample)

            ### print stats about training
            steps_elasped_time += time.time() - step_start_time
            stat = trainer.print_stats(steps_elasped_time)
            if stat:
                steps_elasped_time = 0

            ### write generated images
            trainer.generator_ddp.eval()
            trainer.sample_imgs()
            ### save model parameters
            trainer.save_models()

            ### compute fid score and output generated images
            trainer.evaluate()

            generator.step += 1
            if trainer.metadata["enable_discriminator"]:
                discriminator.step += 1
                assert discriminator.step == generator.step
            # cnt+=1
            # if cnt>100:
            #     return
            if opt.stop_step:
                if generator.step > opt.stop_step:
                    return

        epoch_time_elapsed = time.time() - epoch_start_time
        if rank == 0:
            print(
                f"{generator.epoch}_th epoch runtime: {int(epoch_time_elapsed//60):02}:{int(epoch_time_elapsed%60):02}"
            )
        generator.epoch += 1
        if trainer.metadata["enable_discriminator"]:
            discriminator.epoch += 1
            assert discriminator.epoch == generator.epoch
    if opt.ddp:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train piGAN")
    # parser.add_argument(
    #     "-c",
    #     "--curriculum",
    #     type=str,
    #     default='shapenet',
    #     help="CARLA or CO3D or CATS or CelebA or shapenet",
    # )
    parser.add_argument(
        "-s",
        "--sampling_interval",
        type=int,
        default=200,
        help="step interval between generate sample images",
    )
    parser.add_argument(
        "-p",
        "--print_freq",
        type=int,
        default=100,
        help="step interval between print during training",
    )
    parser.add_argument(
        "-e",
        "--eval_freq",
        type=int,
        default=5000,
        help="step interval between evaluate fid score and save models",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="directory to store and load(in case of reschedule) the output files",
    )
    parser.add_argument(
        "-l",
        "--load_dir",
        type=str,
        default="",
        help="absolute path to the checkpoint (.tar) ",
    )
    parser.add_argument(
        "--load_curriculum",
        type=str,
        default="",
        help="Load existing curriculum. If this argument is specifiec, the trainer will first load curriculum from this path, and will not load curriculum from load_dir/output_dir.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=3000, help="number of epochs of training"
    )
    # parser.add_argument("--port", type=str, default="12355", help="used in setup()")
    parser.add_argument(
        "--stop_step",
        type=int,
        default=None,
        help="Stop training at stop_step.",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="If true, use ddp training; else, normal training on single gpu",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="The specific config to load into the curriculum",
    )
    parser.add_argument(
        "--config_base",
        type=str,
        default="thesis",
        help="The specific config to load into the curriculum",
    )
    # parser.add_argument("--only_test", action="store_true",
    #     help="If true, only test, no train")

    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "logs"), exist_ok=True)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        num_gpus = 1
    print("Number of GPUS: ", num_gpus)
    if num_gpus > 1:
        assert opt.ddp
    print(datetime.now().strftime("%d--%H:%M"))
    print("---------------- Start training ----------------")
    # if opt.only_test:
    #     mp.spawn(test, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    if opt.ddp:
        free_port = find_free_network_port()
        mp.spawn(train, args=(opt, num_gpus, free_port), nprocs=num_gpus, join=True)
    else:
        train(0, opt)
