import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
downsample_ratio = 0.0625
pcl_dirs = glob(
    "/storage/user/yenamand/one/differentiable_volumetric_rendering/data/ShapeNet/02958343/*/pointcloud.npz"
)
print(pcl_dirs[:5])
for dir in tqdm(pcl_dirs):
    pcl_npz = np.load(dir)
    pcl = pcl_npz["points"]
    assert pcl.shape[0] == 100000
    pcl_downsample = pcl[
        np.random.choice(pcl.shape[0], int(pcl.shape[0] * downsample_ratio), replace=False),
        :,
    ]
    np.save(Path(dir).parent / "pointcloud_downsampled", pcl_downsample)
