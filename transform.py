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


class CropByLDK(object):
    def __init__(self, scale):
        assert isinstance(scale, float)
        if isinstance(scale, float):
            self.scale = scale
        else:
            self.scale = 1.4
            
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        left, right = int(landmarks[:,0].min()), int(landmarks[:,0].max())
        top, bottom = int(landmarks[:,1].min()), int(landmarks[:,1].max())
        height, width = int(bottom - top), int(right - left)
        l = int(max(height, width) * self.scale)
            
        middle = [left/2 + right/2, top/2 + bottom/2]
        left_n, right_n = int(middle[0] - l/2), int(middle[0] + l/2)
        top_n, bottom_n = int(middle[1] - l/2), int(middle[1] + l/2)
        if left_n < 0 or right_n > w or top_n < 0 or bottom_n > h:
            blank = np.zeros((l,l,3), dtype=np.float32)
            blank[(l - height)/2 : (l + height)/2, (l - width)/2 : (l + width)/2] = image[top:bottom, left:right]
            image = blank
        else:    
            image = image[top_n:bottom_n, left_n:right_n]
        landmarks = landmarks - [left_n, top_n]
        return {'image': image, 'landmarks':landmarks}
    
class Rescale(object):
    def __init__(self, size):
        assert isinstance(size, tuple)
        self.size = size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        image = transform.resize(image, (self.size[0], self.size[1]))
        landmarks = landmarks * [self.size[0], self.size[1]] / [w, h]
        return {'image': image, 'landmarks':landmarks}
    
class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = transform.rotate(image, self.angle)
        left, right = min(landmarks[:,0]), max(landmarks[:,0])
        top, bottom = min(landmarks[:,1]), max(landmarks[:,1])
        angle = math.radians(-self.angle)
        rotate_matrix = [[math.cos(angle), -math.sin(angle)]
                          ,[math.sin(angle), math.cos(angle)]]
        middle = [left/2 + right/2, top/2 + bottom/2]
        landmarks_r = (landmarks - middle).transpose((1,0))
        landmarks_r = np.matmul(rotate_matrix, landmarks_r)
        landmarks_r = landmarks_r.transpose((1,0))
        landmarks = landmarks_r + middle
        
        return {'image': image, 'landmarks': landmarks}
    
class Flip(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        image = Image.fromarray(np.uint8(image * 256))
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = np.asarray(image)
        image = np.float32(image)/256
        landmarks[:, 0] = w - landmarks[:, 0]
        return {'image' : image, 'landmarks': landmarks}

class RandomCrop(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        r = image.shape[0]
        size = int(r * random.uniform(0.6, 1.0))
        init = r - size
        x0 = random.randint(0, init - 1)
        y0 = random.randint(0, init - 1)
        x1 = x0 + int(size - 1)
        y1 = y0 + int(size - 1)
        image = image[y0:y1, x0:x1]
        landmarks = landmarks - [x0, y0]
        return {'image' : image, 'landmarks': landmarks}
        
class Gnoise(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = skimage.util.random_noise(image,mode='gaussian',seed=None,clip=True)
        return {'image': image, 'landmarks': landmarks}
    
class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}
class Normalize(object):
    def __call__(self,sample):
        image, landmarks = sample['image'], sample['landmarks']
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = normalize(image)
        landmarks = landmarks / image.shape[1]
        image = image.float()
        landmarks = landmarks.float()
        return {'image': image, 'landmarks':landmarks}
        
