import cv2
import torch
import torchvision
import os
import math
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
device = torch.device("cuda:0")

backbone = torchvision.models.mobilenet_v2(pretrained=False).features
backbone.out_channels = 1280
anchor_sizes = ((4, 8, 16, 32, 64, 128, 256, 512),)
aspect_ratios = ((1/2, 1/3, 1/4, 1/5, 1/6, 1/math.sqrt(2), 1,
                  2, math.sqrt(2), 3, 4, 5, 6, 7, 8),)
anchor_generator = AnchorGenerator(
    sizes=anchor_sizes, aspect_ratios=aspect_ratios)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=7,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=None)
model.to(device)


class DocDataset(torch.utils.data.Dataset):
    def __init__(self, root_url, transform=None):
        # self.transform = transform
        self.root_url = root_url
        self.root_fld = os.listdir(self.root_url)
        self.txt_fld = [i for i in self.root_fld if i.split('.')[1] == 'txt']
        self.targets = []
        self.images = []
        for i in self.txt_fld:
            img = cv2.imread('{}/{}.png'.format(root_url, i.split('.')[0]))
            img = torch.tensor(img, dtype=torch.float32) / 255
            img = img.permute(2, 0, 1)
            self.images.append(img)
            d = {}
            boxes = []
            labels = []
            txt = open('{}/{}'.format(root_url, i))
            for j in txt.readlines():
                x0, y0, x1, y1, class_name = j.strip().split(' ')
                boxes.append([int(x0), int(y0), int(x1), int(y1)])
                if class_name == 'send':
                    class_name = 1
                elif class_name == 'number':
                    class_name = 2
                elif class_name == 'date':
                    class_name = 3
                elif class_name == 'quote':
                    class_name = 4
                elif class_name == 'header':
                    class_name = 5
                elif class_name == 'motto':
                    class_name = 6
                labels.append(class_name)
            d['boxes'] = torch.tensor(boxes, dtype=torch.int64)
            d['labels'] = torch.tensor(labels, dtype=torch.int64)
            self.targets.append(d)

    def __len__(self):
        return len(self.txt_fld)

    def __getitem__(self, idx):
        # w,h = self.images[idx]
        return self.images[idx], self.targets[idx]


root_url = '/home/dung/DocData/TB'
dataset = DocDataset(root_url)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0)

for i in range(200):
    print('Epoch {}\n'.format(i))
    for j, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        a = {}
        a['boxes'] = targets['boxes'][0].to(device)
        a['labels'] = targets['labels'][0].to(device)
        # images = list(image for image in images)
        # targets = [{k: v for k, v in t.items()} for t in targets]
        output = model(images, [a])
        if j % 30 == 0:
            print('Step {} -- loss_classifier = {} -- loss_box_reg = {} -- loss_objectness = {} -- loss_rpn_box_reg = {}\n'.format(j,
                                                                                                                                   output['loss_classifier'].item(), output['loss_box_reg'].item(), output['loss_objectness'].item(), output['loss_rpn_box_reg'].item()))
torch.save(model.state_dict(), '1.pth')
print('done')
