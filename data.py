import os
import torch
import pandas as pd
import math
import skimage
import random
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import xml.dom.minidom as xmldom
from scipy.io import loadmat
from transform import *

class Ldk_300W_Dataset(Dataset):
    def __init__(self, xmlfile, root_dir, transform=None):
        document = xmldom.parse(xmlfile)
        annos = document.getElementsByTagName("image")
        self.records = []
        for anno in annos:
            parts = anno.getElementsByTagName("part")
            landmark = []
            for part in parts:
                landmark.append(part.getAttribute("x"))
                landmark.append(part.getAttribute("y"))
            record = [anno.getAttribute("file"), landmark]
            self.records.append(record)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.records[idx][0])
        image = io.imread(img_name)
        h,w = image.shape[:2]
        if len(image.shape) == 2:
            image = image.reshape(h, w, 1)
            image = np.concatenate((image,image,image),axis=2) 
        image = np.float32(image)/256
        landmarks = map(float, self.records[idx][1])
        landmarks = np.asarray(landmarks).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Ldk_300W_LP_Dataset(Dataset):
    def __init__(self, txtfile, root_dir, transform=None):
        f = open(os.path.join(root_dir, txtfile))
        content = f.read()
        content = content.split("\n")[0:-1]
        f.close()
        self.records = []
        for line in content:
            record = line.split(",")
            self.records.append(record)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.records[idx][0])
        image = io.imread(img_name)
        h,w = image.shape[:2]
        if len(image.shape) == 2:
            image = image.reshape(h, w, 1)
            image = np.concatenate((image, image, image), axis=2)
        image = np.float32(image) / 256
        m = loadmat(os.path.join(self.root_dir, self.records[idx][1]))
        landmarks = m['pts_2d']
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch,padding=0)
    plt.imshow(grid.numpy().transpose((1, 2, 0)) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * (im_size),
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')



