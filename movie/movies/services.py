from __future__ import print_function
import argparse
import os
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm

class DcGan(object):


    def __init__(self):
        self.dataroot = './data'
        self.workers = 2
        self.batch_size = 128
        self.image_size = 64
        self.nc = 64
        self.nz = 100
        self.ngf = 64
        self.epochs = 10
        self.lr = 0.0002
        self. beta1 =0.5
        self.ngpu = 1

    def seed(self):
        manualSeed = 999
        # manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def display(self):

        dataset = dset.ImageFolder(root= self.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=self.workers)

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        #Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

GAN_MENUS = ["종료", #0
            "SEED",#1
            'DISPLAY'#2
                ]

gan_menu = {
    "1" : lambda t: t.seed(),
    "2" : lambda t: t.display(),
    "3" : lambda t: print(" ** No Function ** "),
    "4" : lambda t: print(" ** No Function ** "),
    "5" : lambda t: print(" ** No Function ** "),
    "6" : lambda t: print(" ** No Function ** "),
    "7" : lambda t: print(" ** No Function ** "),
    "8" : lambda t: print(" ** No Function ** "),

}
def my_menu(ls):
    for i, j in enumerate(ls):
        print(f"{i}. {j}")
    return input('메뉴선택: ')

if __name__ == '__main__':
    while True:
        t = DcGan()
        menu = my_menu(GAN_MENUS)
        if menu == '0':
            print('종료')
            break
        else:
            gan_menu[menu](t)
