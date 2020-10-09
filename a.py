import cv2
import torch
import torchvision
import os
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import math
import numpy as np
real_img = cv2.imread(
    '/home/dung/Project/Python/keras-frcnn/result/15_19_4.png')
file = open('/home/dung/Project/Python/keras-frcnn/output.txt')
img_file, x0, y0, x1, y1, class_name = file.readlines()[0].strip().split(',')
cv2.rectangle(real_img, (int(x0), int(y0)),
              (int(x1), int(y1)), (100, 200, 150), 1, 1)
cv2.imshow('aaa', real_img)
cv2.waitKey(0)
