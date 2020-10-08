import cv2
import torch
import torchvision
import os
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import math
device = torch.device("cuda:0")
backbone = torchvision.models.mobilenet_v2(pretrained=False).features
backbone.out_channels = 1280

anchor_sizes = ((4, 8, 16, 32, 64, 128, 256, 512),)
aspect_ratios = ((1/2, 1/3, 1/4, 1/5, 1/6, 1/math.sqrt(2), 1,
                  2, math.sqrt(2), 3, 4, 5, 6, 7, 8),)
anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=7,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=None)
model.load_state_dict(torch.load('1.pth'))
model.eval()
model.to(device)
real_img = cv2.imread('/home/dung/DocData/cp/145/110.png')
img = torch.tensor(real_img, dtype=torch.float32)/255
img = img.permute((2, 0, 1))

output = model([img.to(device)])
print(output)
