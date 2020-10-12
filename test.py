import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
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
                   box_score_thresh=0.5,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler,
                   box_detections_per_img=20)

model.to(device)


class DocDataset(torch.utils.data.Dataset):
    def __init__(self, root_url):
        # self.transform = transform
        self.root_url = root_url
        self.root_fld = root_url.rsplit('/', 1)[0]
        file = open(self.root_url)
        self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        self.images = []
        A, y0, x1, y1, class_name = self.lines[idx].strip().split(
            ',')
        img_file, x0 = A.strip().split(' ')
        img = cv2.imread('{}/{}'.format(self.root_fld, img_file))
        # cv2.rectangle(img, (int(x0), int(y0)),
        #       (int(x1), int(y1)), (100, 200, 150), 1, 1)
        # cv2.imshow('aaa', img)
        # cv2.waitKey(0)
        img = torch.tensor(img, dtype=torch.float32) / 255
        img = img.permute(2, 0, 1)
        # self.images.append(img)
        boxes = [(int(x0), int(y0), int(x1), int(y1))]
        labels = [1]
        targets = {'boxes': torch.tensor(boxes, dtype=torch.int64),
                   'labels': torch.tensor(labels, dtype=torch.int64)}

        return img, targets


root_url = '/home/dung/Project/Test/faster-torch/output.txt'
dataset = DocDataset(root_url)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for i in range(500):
    print('Epoch {}\n'.format(i))
    for j, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        a = {}
        a['boxes'] = targets['boxes'][0].to(device)
        a['labels'] = targets['labels'][0].to(device)
        output = model(images, [a])
        losses = sum(loss for loss in output.values())
        if j % 300 == 0:
            print('Step {} -- loss_classifier = {} -- loss_box_reg = {} -- loss_objectness = {} -- loss_rpn_box_reg = {}\n'.format(j,
                                                                                                                                   output['loss_classifier'].item(), output['loss_box_reg'].item(), output['loss_objectness'].item(), output['loss_rpn_box_reg'].item()))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    if i % 50 == 0:
        torch.save(model.state_dict(), '1.pth')
print('done')
