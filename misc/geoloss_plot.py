import torch
import sys
sys.path.append('../')
from utils import sample_negative_points, jsonKeys2int, encode_pcl, loss_geometry, loss_mse
from pathlib import Path
from generators import siren, generators
from generators.pointnet_encoder import ResnetPointnet
import json
from configs import curriculums
from train import get_dataloader
from collections import defaultdict
from matplotlib import pyplot as plt
import pickle

ck_paths = '/storage/slurm/zhouzh/geoloss_unfreeze/photo_loss_1000_2/checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dicts = defaultdict(dict)
for dir in Path(ck_paths).iterdir():
    step = int(dir.stem)
    ckpt=torch.load(dir, map_location=device)
    state_dicts[step]['generator']= ckpt['generator_state_dict']
    state_dicts[step]['encoder']= ckpt['encoder_state_dict']
    assert step == ckpt['step']


SIREN = siren.TALLSIREN
generator = generators.ImplicitGenerator3d(SIREN, 512, 0, "CustomMappingNetwork")
generator.to(device)
generator.set_device(device)
encoder = ResnetPointnet()

with open(Path(ck_paths).parent / 'curriculum.json', 'r') as f:
    curriculum = json.load(f, object_hook=jsonKeys2int)
curriculum["nerf_noise"] = 0

geo_loss_dict = {}
photo_loss_dict = {}
batches_to_sample = 10
with torch.no_grad():
    for step, state_dict in state_dicts.items():
        metadata = curriculums.extract_metadata(curriculum, step)
        dataloader = get_dataloader(metadata, False, 1, 0)
        encoder.load_state_dict(state_dict['encoder'])
        encoder.to(device)
        generator.load_state_dict(state_dict['generator'])
        generator.step = step
        generator.to(device)
        generator.eval()
        geo_loss = 0
        photo_loss = 0
        for idx, (imgs, pcl, cam) in enumerate(dataloader):
            imgs=imgs.to(device)
            pcl_pos = pcl[
                        :,
                        torch.randperm(100000)[
                            : int(100000 * metadata["num_points_ratio"])
                        ],
                    ]
            z = encode_pcl(encoder, pcl_pos, device, 0)
            pcl_all, num_pos_points = sample_negative_points(
                    pcl_pos,
                    metadata["kdtree_r"],
                )
            pcl_all = pcl_all.to(device)
            gen_imgs, gen_positions, sigma_preds = generator(
                        z,
                        pcl=pcl_all,
                        cam_origin_data=cam,
                        **metadata,
                    )
            geo_loss+=loss_geometry(sigma_preds, num_pos_points)
            photo_loss+=loss_mse(imgs, gen_imgs)
            # print(geo_loss/(idx+1))
            # print(photo_loss/(idx+1))
            if idx>=batches_to_sample-1:
                break
        geo_loss_dict[step] = geo_loss.item()/batches_to_sample
        photo_loss_dict[step]=photo_loss.item()/batches_to_sample
        print(geo_loss_dict)
        print(photo_loss_dict)
result = {}
result['geo'] = geo_loss_dict
result['photo'] = photo_loss_dict

with open('loss_geo', 'wb') as f:
    pickle.dump(result, f)




        





