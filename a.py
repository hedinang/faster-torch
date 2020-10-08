import cv2
import torch
import torchvision
import os
import numpy as np
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import math
device = torch.device("cuda:0")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
real_img = cv2.imread('/home/dung/Image/2.jpeg')
img = torch.tensor(real_img, dtype=torch.float32)/255
img = img.permute((2, 0, 1))
model.eval()
model.to(device)
output = model([img.to(device)])
boxes = output[0]['boxes']
a = output[0]['boxes'].detach().to('cpu').numpy()
a = np.round(a)
print(output)
for (x0, y0, x1, y1) in a[:1]:
    cv2.rectangle(real_img, (x0, y0), (x1, y1), (100, 200, 150), 1, 1)
cv2.imshow('aaa',real_img)
cv2.waitKey(0)
