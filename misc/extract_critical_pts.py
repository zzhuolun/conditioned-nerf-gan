import torch
import numpy as np
from generators.pointnet_encoder import ResnetPointnet
from pathlib import Path
from misc.npy2obj import npy2obj
import argparse


def extract_critical_points(
    pcl_path: str, ckpt_path: str, save_dir: str, dim: int
) -> None:
    pcl_np = np.load(pcl_path)
    pcl = torch.from_numpy(pcl_np).type(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    encoder = ResnetPointnet(dim=dim)
    encoder.to(device)
    encoder.load_state_dict(ck["encoder_state_dict"])
    pcl = pcl.to(device)
    with torch.no_grad():
        z, (idx0, idx1, idx2, idx3, idx4) = encoder.forward_with_idx(pcl.unsqueeze(0))

    pcl_c0 = pcl_np[idx0[0, 0].cpu().numpy()]
    pcl_c1 = pcl_np[idx1[0, 0].cpu().numpy()]
    pcl_c2 = pcl_np[idx2[0, 0].cpu().numpy()]
    pcl_c3 = pcl_np[idx3[0, 0].cpu().numpy()]
    pcl_c4 = pcl_np[idx4[0].cpu().numpy()]

    # TODO: consider pcl has no color

    pcl_c0[:, 3:] = np.ones_like(pcl_c0[:, 3:]) * np.array([[0, 0, 0]])
    pcl_c1[:, 3:] = np.ones_like(pcl_c1[:, 3:]) * np.array([[1, 1, 1]])
    pcl_c2[:, 3:] = np.ones_like(pcl_c2[:, 3:]) * np.array([[0, 0, 1]])
    pcl_c3[:, 3:] = np.ones_like(pcl_c3[:, 3:]) * np.array([[0, 1, 0]])
    pcl_c4[:, 3:] = np.ones_like(pcl_c4[:, 3:]) * np.array([[1, 0, 0]])

    pcl_c = np.concatenate([pcl_c0, pcl_c1, pcl_c2, pcl_c3, pcl_c4], axis=0)
    npy2obj(pcl_c, Path(save_dir) / (Path(pcl_path).parent.stem + "_c.obj"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pcl_path",
        type=str,
        help="absolute path to the .npy points file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="path to the saved checkpoint",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory to save the .obj file",
    )
    parser.add_argument("--colored_pcl", action="store_true")
    opt = parser.parse_args()
    dim = 6 if opt.colored_pcl else 3
    extract_critical_points(opt.pcl_path, opt.ckpt_path, opt.save_dir, dim)
