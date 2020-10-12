import torch
from craft import CRAFT
from imgproc import resize_aspect_ratio, normalizeMeanVariance
from craft_ultils import getDetBoxes, adjustResultCoordinates
from utils import group_text_box, get_image_list, calculate_md5, get_paragraph,\
    download_and_unzip, printProgressBar, diff, reformat_input
import cv2
from torch.autograd import Variable
from collections import OrderedDict
from modules import vgg16_bn, init_weights
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
import time
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
# predict = Predictor(config)
# time1 = time.time()
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


device = torch.device('cpu')
model = CRAFT()
model.load_state_dict(copyStateDict(torch.load(
    '/home/dung/Project/Python/ocr/craft_mlt_25k.pth')))
model.to(device)
model.eval()
model.share_memory()


def detect(model, image):
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 2560,
                                                                  interpolation=cv2.INTER_LINEAR, mag_ratio=1.)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))               # [c, h, w] to [b, c, h, w]
    x = x.to(device)
    # with torch.no_grad():
    y, feature = model(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link,
                               text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    horizontal_list, free_list = group_text_box(result, slope_ths=0.8,
                                                ycenter_ths=0.5, height_ths=1,
                                                width_ths=1, add_margin=0.1)
    # horizontal_list = [i for i in horizontal_list if i[0] > 0 and i[1] > 0]
    min_size = 20
    if min_size:
        horizontal_list = [i for i in horizontal_list if max(
            i[1]-i[0], i[3]-i[2]) > 10]
        free_list = [i for i in free_list if max(
            diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]

    sub_img_list = []
    # for ele in horizontal_list:
    #     ele = [0 if i < 0 else i for i in ele]
    #     sub_img = img_resized[ele[2]:ele[3], ele[0]:ele[1], :]
    #     img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(img)
    #     s = predict.predict(img)
    #     sub_img_list.append(s)
    #     sub_img_list.append(sub_img)
    # print(time.time() - time1)
    # return sub_img_list


for i in range(3):

    image = cv2.imread('{}.png'.format(i))
    p = mp.Process(target=detect, args=(model, image,))
    p.start()
    p.join()
print('last {}'.format(time.time() - time1))
print('aaa')
