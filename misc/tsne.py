from sklearn import datasets
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
sys.path.append('../')
from datasets import _read_resize_shapenet
from datetime import datetime

def custom_color():
    tableau20 = [ # with 10 more customized color
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
        (0, 0, 0),
        (65, 68, 81),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (96, 99, 106),
        (210,210,210),
        (0, 107, 164),
        (219, 161, 58),
        (143, 135, 130),
        (153, 86, 136),

    ]

    # Tableau Color Blind 10
    tableau20blind = [
        (0, 107, 164),
        (255, 128, 14),
        (171, 171, 171),
        (89, 89, 89),
        (95, 158, 209),
        (200, 82, 0),
        (137, 137, 137),
        (163, 200, 236),
        (255, 188, 121),
        (207, 207, 207),
    ]

    # Rescale to values between 0 and 1
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)
    for i in range(len(tableau20blind)):
        r, g, b = tableau20blind[i]
        tableau20blind[i] = (r / 255.0, g / 255.0, b / 255.0)
    return tableau20, tableau20blind


class ShapeNet(Dataset):
    """ShapeNet Car Dataset"""

    def __init__(self, dataset_path, num_cars, img_size=224, random_cars=True):
        super().__init__()
        cars_dir_ls = [i for i in Path(dataset_path).iterdir() if i.is_dir()]
        if random_cars:
            cars_path = random.sample(cars_dir_ls, num_cars)
        else:
            cars_path = cars_dir_ls[:num_cars]
        self.data = []
        for p in cars_path:
            self.data += list((p / "image").iterdir())
        self.transform = transforms.Compose(
            [
                # transforms.Resize((img_size, img_size), interpolation=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        X = _read_resize_shapenet(str(img_path), self.img_size)
        X = self.transform(X)

        return X, img_path.parent.parent.stem


class TsneCar():
    
    """Visualize the resnet encoding of shapenet cars by tSNE"""
    def __init__(self, dataset_path, num_cars, tsne_dim, tsne_perplexity, pca_dim=50):
        self.dataset_path = dataset_path
        print(f"Loading {num_cars} cars...")
        dataset = ShapeNet(dataset_path, num_cars=num_cars, random_cars=False)
        self.dataloader = DataLoader(dataset, batch_size=8, num_workers=2)
        self.num_cars = num_cars

        self.tsne = TSNE(n_components=tsne_dim, perplexity=tsne_perplexity, learning_rate="auto", init="random")
        self.pca = PCA(n_components=pca_dim)

        resnet = resnet18(pretrained=True, progress=False).eval()
        resnet.fc = nn.Sequential()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resnet = resnet.to(self.device)

    def _reduce_dim(self):
        """ Img [224x224]  -- ResNet18 --> encoding [512] -- PCA --> reduced dim to [50] -- tSNE --> further reduce dim to [2] for visualization """
        # colors
        col = custom_color()[0]
        car_color = []
        for i in range(self.num_cars * 24):
            car_color.append(col[i // 24])

        # get encoding
        car_name = []
        car_data = []
        with torch.no_grad():
            for X, car in self.dataloader:
                X = X.to(self.device)
                y = self.resnet(X)
                for idx, c in enumerate(car):
                    code = y[idx].cpu().numpy()
                    # normalize the encoding output
                    code -= code.mean()
                    code /= code.std()
                    car_data.append(code)
                    car_name.append(c)
        print("Car sequence names: \n", set(car_name))

        # assert same car mapped to same color
        color_mapping = []
        for idx, c in enumerate(car_name):
            color_mapping.append((c, car_color[idx]))
        color_mapping = set(color_mapping)
        assert len(color_mapping) == self.num_cars, "plot color does not map cars"

        # dimension reduction
        car_pca = self.pca.fit_transform(car_data)
        car_tsne = self.tsne.fit_transform(car_pca)
        print("t-SNE output shape: ", car_tsne.shape)
        return car_tsne, car_color, color_mapping

    def visualize(self, save_name):
        car_tsne, car_color, color_mapping = self._reduce_dim()
        fig1 = plt.figure(figsize=(12, 12))
        plt.scatter(car_tsne[:, 0], car_tsne[:, 1], c=car_color)
        fig1.suptitle(f't-SNE of {self.num_cars} shapenet cars encoded by ResNet18 (encoding dim=512)', y=0.93, fontsize='xx-large')
        plt.savefig('tmp1.png')
        plt.clf()

        fig2 = plt.figure(figsize=(12,12))
        grid = ImageGrid(fig2, 111,  # similar to subplot(111)
                 nrows_ncols=(5, int(self.num_cars/5)),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
        color_size = 10
        images = []
        for name,color in color_mapping:
            img = _read_resize_shapenet(str(Path(self.dataset_path) / name / 'image' / '0000.png'), 64) / 255
            img[:color_size,:color_size,:] = color
            images.append(img)

        for ax, im in zip(grid, images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
        fig2.suptitle('color legend: color in the top-left corner of each car refers to the color in tSNE plot', y=0.93, fontsize='xx-large')
        plt.savefig('tmp2.png')

        os.system(f'ffmpeg -i tmp1.png -i tmp2.png -filter_complex hstack {save_name}')
        os.system('rm tmp1.png')
        os.system('rm tmp2.png')


if __name__ == "__main__":
    path = "/usr/stud/zhouzh/data/ShapeNetCar/"
    tsne_dim = 2
    tsne_perplexity = 10
    num_cars = 20
    tsne_car = TsneCar(path,num_cars=num_cars, tsne_dim=tsne_dim, tsne_perplexity=tsne_perplexity)
    timestamp = datetime.now().strftime("%d--%H:%M")
    tsne_car.visualize(f"archive/tsne_{tsne_perplexity}-{timestamp}.png")