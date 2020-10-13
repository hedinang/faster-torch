import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np


class Detect:
    def __init__(self):
        super().__init__()
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
        self.model = FasterRCNN(backbone,
                                num_classes=7,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler)
        self.device = torch.device('cpu')
        self.model.load_state_dict(torch.load('2.pth'))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, img):
        img = torch.tensor(img, dtype=torch.float32)/255
        img = img.permute((2, 0, 1))
        output = model([img.to(self.device)])
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        last = {}
        result = {}
        for i, v in enumerate(labels):
            if v == 1 and scores[i] > last['send']:
                last['send'] = scores[i]
                result['send'] = boxes[i]
            elif v == 2 and scores[i] > last['number']:
                last['number'] = scores[i]
                result['number'] = boxes[i]
            elif v == 3 and scores[i] > last['date']:
                last['date'] = scores[i]
                result['date'] = boxes[i]
            elif v == 4 and scores[i] > last['quote']:
                last['quote'] = scores[i]
            elif v == 5 and scores[i] > last['header']:
                last['header'] = scores[i]
                result['header'] = boxes[i]
            elif v == 6 and scores[i] > last['motto']:
                last['motto'] = scores[i]
                result['motto'] = boxes[i]
            # elif v == 7 and scores[i] > last['secrete']:
            #     last['secrete'] = scores[i]
            #     result['secrete'] = boxes[i]
            # elif v == 8 and scores[i] > last['sign']:
            #     last['sign'] = scores[i]
            #     result['sign'] = boxes[i]
        return result
