import torch
import torchvision
import cv2
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
device = torch.device("cuda:0")
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
print(model)
# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 7  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

root_url = '/home/dung/DocData/test'
root_fld = os.listdir(root_url)
txt_fld = [i for i in root_fld if i.split('.')[1] == 'txt']
images = []
targets = []
epochs = 10

for i in txt_fld:
    img = cv2.imread('{}/{}.png'.format(root_url, i.split('.')[0]))
    img = torch.tensor(img, dtype=torch.float32)/255
    img = img.permute((2, 0, 1))
    images.append(img)
    f = open('{}/{}'.format(root_url, i))
    boxes = []
    labels = []
    for j in f.readlines():
        x0, y0, x1, y1, class_name = j.strip().split(' ')
        boxes.append([int(x0), int(y0), int(x1), int(y1)])
        if class_name == 'send':
            labels.append(1)
        elif class_name == 'date':
            labels.append(2)
        elif class_name == 'quote':
            labels.append(3)
        elif class_name == 'header':
            labels.append(4)
        elif class_name == 'motto':
            labels.append(5)
        elif class_name == 'number':
            labels.append(6)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    d = {}
    d['boxes'] = boxes
    d['labels'] = labels
    targets.append(d)
# images = images.to(device)
# target['']
for epoch in range(epochs):
    output = model(images, targets)
    print(output)
