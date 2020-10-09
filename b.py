import cv2
import torch
import torchvision
import os
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import math
import numpy as np
device = torch.device('cuda:0')

backbone = torchvision.models.vgg16(pretrained=False).features
backbone.out_channels = 512
anchor_sizes = ((8, 16, 32, 64, 128, 256, 512),)
aspect_ratios = ((1/2, 1/3, 1/4, 1/5, 1/6, 1/math.sqrt(2), 1,
                  2, math.sqrt(2), 3, 4, 5, 6, 7, 8),)
anchor_generator = AnchorGenerator(
    sizes=anchor_sizes, aspect_ratios=aspect_ratios)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
model.load_state_dict(torch.load('1.pth'))
model.to(device)
model.eval()
real_img = cv2.imread(
    '/home/dung/Project/Python/keras-frcnn/result/0_19_0.png')
img = torch.tensor(real_img, dtype=torch.float32)/255
img = img.permute((2, 0, 1))

output = model([img.to(device)])

boxes = output[0]['boxes']
a = output[0]['boxes'].detach().to('cpu').numpy()
a = np.round(a)
print(output)
for (x0, y0, x1, y1) in a[:3]:
    cv2.rectangle(real_img, (x0, y0), (x1, y1), (100, 200, 150), 1, 1)
cv2.imshow('aaa', real_img)
cv2.waitKey(0)
