import argparse
import copy
from typing import Union
from pathlib import Path
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def plot_mrc(path:Union[Path, str], thres:float)->None:
    """Plot the voxel from saved .mrcfile

    Args:
        path (Union[Path, str]): Absolute path to the voxel file
        thres (float): The density threshold. Only voxel density higher than the threshold will be plotted.
    """
    f = mrcfile.open(path)
    v = copy.deepcopy(f.data)
    print('max sigma', v.max())
    print('min sigma', v.min())
    v[np.where(v<thres)]=0
    ax = plt.figure().add_subplot(projection="3d")
    ax.voxels(v)
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--thres", type=float, default=0) 
    opt = parser.parse_args()
    plot_mrc(opt.path, opt.thres)


