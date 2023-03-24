# -------------------------------------
# Draw the loss graph of experiments.
# -------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import argparse
from utils import _find_newest_ckpoints
from typing import Union


def _moving_average_window(x: np.ndarray, window_len: int) -> np.ndarray:
    if window_len >= 1:
        win = np.ones(window_len) / window_len
        return np.convolve(x, win, mode="valid")
    else:
        return x


def draw_multiple_losses(dirs: list[Path], window_len: int, save: bool = False) -> None:
    """Draw and compare losses of multiple experiment

    Args:
        dir (list[Path]): the directores to the experiments
        window_len (int): the lenght of window used to smoothe the loss
        save (bool, optional): If true, the loss plot will be saved. Defaults to False.
    """
    ck_names = [
        i / "checkpoints" / _find_newest_ckpoints(i / "checkpoints") for i in dirs
    ]
    checkpoints = [
        torch.load(ck_name, map_location=torch.device("cpu")) for ck_name in ck_names
    ]
    possible_losses = {
        "discriminator_losses": True,
        "generator_losses": True,
        "geometry_losses": True,
        "photometry_losses": True,
        "depth_losses": True,
        "photometry_losses_test": True,
        "depth_losses_test": True,
        "photometry_losses_val": True,
        "depth_losses_val": True,
    }
    for l in possible_losses.keys():
        for c in checkpoints:
            if not (l in c.keys() and c[l]):
                possible_losses[l] = False
    title = [l for l, v in possible_losses.items() if v]
    num_losses = len(title)
    if num_losses == 0:
        print("No losses to compare!")
        return
    fig, axes = plt.subplots(num_losses, sharex=True, sharey=True, figsize=(8, 6))
    if num_losses == 1:
        axes = [axes]
    fig.suptitle("Loss comparisions")
    plt.text(
        x=0.8,
        y=0.9,
        s=f"Window length: {window_len}",
        fontsize=8,
        ha="center",
        transform=fig.transFigure,
    )
    labels = [i.stem for i in dirs]
    for i in range(num_losses):
        losses = [c[title[i]] for c in checkpoints]
        for idx, l in enumerate(losses):
            if title[i] not in [
                "photometry_losses_test",
                "depth_losses_test",
                "photometry_losses_val",
                "depth_losses_val",
            ]:
                axes[i].plot(
                    _moving_average_window(np.asarray(l[1:]), window_len),
                    label=labels[idx],
                )
            else:
                l = np.asarray(l)
                axes[i].plot(l[:, 0], l[:, 1], label=labels[idx])
        axes[i].set_title(title[i])
        if i == (num_losses - 1):
            axes[i].legend(loc="best")
        axes[i].grid()
    plt.yscale('log')
    if save:
        print("Saving loss plot to", str(dirs[0] / "loss.jpg"))
        plt.savefig(dirs[0] / "loss.jpg")
    plt.show()


def draw_single_losses(dir: Path, window_len: int, save: bool = False) -> None:
    """Draw loss plot of a single experiment

    Args:
        dir (Path): the directory to the experiment
        window_len (int): the lenght of window used to smoothe the loss
        save (bool, optional): If true, the loss plot will be saved. Defaults to False.
    """
    ck_name = dir / "checkpoints" / _find_newest_ckpoints(dir / "checkpoints")
    checkpoint = torch.load(ck_name, map_location=torch.device("cpu"))
    losses = {}
    possible_losses = [
        "discriminator_losses",
        "generator_losses",
        "geometry_losses",
        "photometry_losses",
        "depth_losses",
        "photometry_losses_test",
        "depth_losses_test",
        "photometry_losses_val",
        "depth_losses_val",
    ]

    for l in possible_losses:
        if (l in checkpoint.keys()) and checkpoint[l]:
            if l not in [
                "photometry_losses_test",
                "depth_losses_test",
                "photometry_losses_val",
                "depth_losses_val",
            ]:
                losses[l] = _moving_average_window(
                    np.asarray(checkpoint[l][1:]), window_len
                )
            else:
                losses[l] = np.asarray(checkpoint[l])

    print("Found losses: ", losses.keys(), "from ", str(ck_name))
    num_losses = len(losses)
    fig, axes = plt.subplots(num_losses, sharex=True, sharey=True, figsize=(8, 6))
    fig.suptitle(dir.name)
    plt.text(
        x=0.8,
        y=0.9,
        s=f"Window length: {window_len}",
        fontsize=8,
        ha="center",
        transform=fig.transFigure,
    )
    if num_losses == 1:
        axes.plot(losses[losses.keys()[0]])
        axes.set_title(losses.keys()[0])
        axes.grid()
    else:
        for i, (title, loss) in enumerate(losses.items()):
            if title in [
                "photometry_losses_test",
                "depth_losses_test",
                "photometry_losses_val",
                "depth_losses_val",
            ]:
                axes[i].plot(loss[:, 0], loss[:, 1])
            else:
                axes[i].plot(loss)
            axes[i].set_title(title)
            axes[i].grid()
    plt.subplots_adjust(hspace=0.6)
    plt.yscale('log')
    # fig.tight_layout()
    if save:
        print("Saving loss plot to", str(dir / "loss.jpg"))
        plt.savefig(dir / "loss.jpg")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw Loss Graph")
    parser.add_argument(
        "dirs",
        nargs="+",
        type=str,
        help="absolute paths to the experiment directories (experiment name)",
    )
    parser.add_argument(
        "-w", "--window_len", help="window length", type=int, default=200
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="if true, save the plot; else, just show the loss plot",
    )

    opt = parser.parse_args()
    experiment_dirs = [Path("/storage/slurm/zhouzh/") / i for i in opt.dirs]
    # experiment_dirs = [Path(i) for i in opt.dirs]
    for dir in experiment_dirs:
        assert dir.exists(), f"{str(dir)} does not exists"
    if len(experiment_dirs) == 1:
        draw_single_losses(experiment_dirs[0], opt.window_len, opt.save)
    else:
        draw_multiple_losses(experiment_dirs, opt.window_len, opt.save)
    # draw_losses(opt.dir, opt.window_len, opt.save)
