import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plotfid(dirs:list):

    fig, axes = plt.subplots(4, sharex=True)
    fig.suptitle("metrics")
    for fid_dir in dirs:
        with open(fid_dir, 'r') as f:
            lines = f.readlines()
        step = []
        fid = []
        ofid = []
        lpips  = []
        psnr = []
        for line in lines[1:]:
            ls = line.split(' ')
            ls = [float(i.strip('\n')) for i in ls]
            step.append(int(ls[0]))
            fid.append(ls[1])
            ofid.append(ls[2])
            lpips.append(ls[3])
            psnr.append(ls[4])

        axes[0].plot(step, fid, label=Path(fid_dir).parent.stem)
        axes[0].set_title("fid")
        axes[1].plot(step, ofid, label=Path(fid_dir).parent.stem)
        axes[1].set_title("ofid")
        axes[2].plot(step, lpips, label=Path(fid_dir).parent.stem)
        axes[2].set_title("lpips")
        axes[3].plot(step, psnr, label=Path(fid_dir).parent.stem)
        axes[3].set_title("psnr")
        axes[3].legend(loc="best")
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs="+", help="relative path to the fid.txt")
    opt=parser.parse_args()
    dirs = [str(Path('/storage/slurm/zhouzh/') / p / 'fid.txt') for p in opt.path]
    plotfid(dirs)