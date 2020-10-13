import torch
from craft import CRAFT
from vietocr.tool.translate import build_model, translate_beam_search, process_input, predict
from imgproc import resize_aspect_ratio, normalizeMeanVariance
from craft_ultils import getDetBoxes, adjustResultCoordinates
from utils import group_text_box
import cv2
from collections import OrderedDict
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method, Value, Manager
import time
from PIL import Image
from vietocr.tool.config import Cfg
import threading


class Ocr:
    def __init__(self):
        super().__init__()
        manager = Manager()
        self.send = manager.list()
        self.date = manager.list()
        self.quote = manager.list()
        self.number = manager.list()
        self.header = manager.list()
        self.sign = manager.list()
        self.device = torch.device('cpu')
        state_dict = torch.load(
            '/home/dung/Project/Python/ocr/craft_mlt_25k.pth')
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v

        self.craft = CRAFT()
        self.craft.load_state_dict(new_state_dict)
        self.craft.to(self.device)
        self.craft.eval()
        self.craft.share_memory()
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        self.config['device'] = 'cpu'
        self.config['predictor']['beamsearch'] = False
        self.weights = '/home/dung/Documents/transformerocr.pth'
        self.model, self.vocab = build_model(self.config)
        self.model.load_state_dict(torch.load(self.weights))
        self.model.to(self.device)
        self.model.eval()
        self.model.share_memory()
        # self.model, self.vocab = build_model(self.config)

    def predict(self, model, vocab, seq, key, idx, img):

        img = process_input(img, self.config['dataset']['image_height'],
                            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
        img = img.to(self.config['device'])
        with torch.no_grad():
            src = model.cnn(img)
            memory = model.transformer.forward_encoder(src)
            translated_sentence = [[1]*len(img)]
            max_length = 0
            while max_length <= 128 and not all(np.any(np.asarray(translated_sentence).T == 2, axis=1)):
                tgt_inp = torch.LongTensor(translated_sentence).to(self.device)
                output = model.transformer.forward_decoder(tgt_inp, memory)
                output = output.to('cpu')
                values, indices = torch.topk(output, 5)
                indices = indices[:, -1, 0]
                indices = indices.tolist()
                translated_sentence.append(indices)
                max_length += 1
                del output
            translated_sentence = np.asarray(translated_sentence).T
        s = translated_sentence[0].tolist()
        s = vocab.decode(s)
        seq[idx] = s
        # print(time.time() - time1)

    def process(self, craft, model, seq, key, sub_img):
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(sub_img, 2560,
                                                                      interpolation=cv2.INTER_LINEAR, mag_ratio=1.)
        ratio_h = ratio_w = 1 / target_ratio

        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0)              # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)
        y, feature = craft(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, polys = getDetBoxes(score_text, score_link,
                                   text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)
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
        seq[:] = [None] * len(horizontal_list)

        for i, ele in enumerate(horizontal_list):
            ele = [0 if i < 0 else i for i in ele]
            img = sub_img[ele[2]:ele[3], ele[0]:ele[1], :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype(np.uint8))
            img = process_input(img, self.config['dataset']['image_height'],
                                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
            img = img.to(self.config['device'])
            with torch.no_grad():
                src = model.cnn(img)
                memory = model.transformer.forward_encoder(src)
                translated_sentence = [[1]*len(img)]
                max_length = 0
                while max_length <= 128 and not all(np.any(np.asarray(translated_sentence).T == 2, axis=1)):
                    tgt_inp = torch.LongTensor(
                        translated_sentence).to(self.device)
                    output = model.transformer.forward_decoder(tgt_inp, memory)
                    output = output.to('cpu')
                    values, indices = torch.topk(output, 5)
                    indices = indices[:, -1, 0]
                    indices = indices.tolist()
                    translated_sentence.append(indices)
                    max_length += 1
                    del output
                translated_sentence = np.asarray(translated_sentence).T
            s = translated_sentence[0].tolist()
            s = self.vocab.decode(s)
            seq[idx] = s

    def forward(self, img, rs):
        # time1 = time.time()
        for key, v in rs.items():
            x0, y0, x1, y1 = v
            if key == 'send':
                p = mp.Process(target=self.process, args=(
                    self.craft, self.send, key, img[y0:y1, x0:x1, :],))
            elif key == 'date':
                p = mp.Process(target=self.process, args=(
                    self.craft, self.date, key, img[y0:y1, x0:x1, :],))
            elif key == 'quote':
                p = mp.Process(target=self.process, args=(
                    self.craft, self.date, key, img[y0:y1, x0:x1, :],))
            elif key == 'number':
                p = mp.Process(target=self.process, args=(
                    self.craft, self.date, key, img[y0:y1, x0:x1, :],))
            elif key == 'header':
                p = mp.Process(target=self.process, args=(
                    self.craft, self.date, key, img[y0:y1, x0:x1, :],))
            elif key == 'sign':
                p = mp.Process(target=self.process, args=(
                    self.craft, self.model, self.date, key, img[y0:y1, x0:x1, :],))
            p.start()
            p.join()
        return self.send[:], self.date[:], self.quote[:], self.number[:], self.header[:], self.sign[:]


ocr = Ocr()
img = cv2.imread('16.png')
result = {}
result['send'] = [210, 196, 691, 279]
result['date'] = [252, 288, 650, 331]
# a = Ocr()
# time1 = time.time()
# send, date, quote, number, header, sign = a.forward(img, result)
# print(time.time() - time1)
rr = ocr.forward(img, result)
print('aaa')
