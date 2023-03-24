# Find the closes latent code to the tart_pcl in the twenty cars for the 20car_color_pcl model in terms of l2 distance
import torch
import numpy as np
from generators.pointnet_encoder import ResnetPointnet
from pathlib import Path
import os
import matplotlib.pyplot as plt


target_pcl_dir = '504793ed2da6cf7eba3e2415e22cd45c'
with open('/usr/stud/zhouzh/data/shapenet_twenty.lst' ,'r') as f:
    pcl_dirs = f.readlines()
pcl_dirs = [i[:-1] for i in pcl_dirs]
pcl_dirs.append(target_pcl_dir)


ckpt_path = '/storage/slurm/zhouzh/20cars_colorpcl/checkpoints/34999.tar'
dim = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(ckpt_path, map_location=device)
encoder = ResnetPointnet(dim=dim)
encoder.to(device)
encoder.load_state_dict(ck["encoder_state_dict"])

z_ls = []
for pcl_path in pcl_dirs:
    pcl_path = os.path.join('/usr/stud/zhouzh/data/ShapeNetCar', pcl_path, 'pcl_color.npy')
    pcl_np = np.load(pcl_path)
    pcl = torch.from_numpy(pcl_np).type(torch.float32)
    pcl = pcl.to(device)
    with torch.no_grad():
        z_ls.append(encoder(pcl.unsqueeze(0))[0].cpu().numpy())
unseen = z_ls[-1]
dist = []
for i in range(20):
    dist.append(np.linalg.norm((z_ls[i] - unseen)))
car_min = pcl_dirs[dist.index(min(dist))]
# img = plt.imread(os.path.join('/usr/stud/zhouzh/data/ShapeNetCar', car_min, 'image/0000.png'))
# plt.imshow(img)
# plt.show()
print(car_min)