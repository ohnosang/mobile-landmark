import torch
import numpy as np
import cv2
from model import *
import torchvision.transforms
cap = cv2.VideoCapture(0);
weights = torch.load('/home/whale/landmark.pth')['net']
model = MobileNetV2()
model.load_state_dict(weights)
while(True):
	flag, img1 = cap.read()
	img = cv2.resize(img1, (64,64))
	image = np.float32(img)/256
	image = image.transpose((2,0,1))

	inputs = torch.Tensor(image)
	normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        inputs = normalize(inputs)
	inputs = inputs.view(-1,3,64,64)
	outputs = model(inputs)
	outputs = (outputs.detach().numpy().reshape([68,2]) * 64).astype(int)

	for i in range(68):
		print outputs[i]
		cv2.circle(img,tuple(outputs[i]),2,(255,255,255),1)
	
	cv2.imshow('frame', img)	
	cv2.waitKey(1)
