from vietocr.tool.translate import build_model, translate_beam_search, process_input, predict
import torch
from vietocr.tool.utils import download_weights
import time
import torch.multiprocessing as mp
import cv2
from vietocr.tool.config import Cfg
from PIL import Image
import numpy as np
import threading
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
device = 'cpu'

model, vocab = build_model(config)
weights = '/home/dung/Documents/transformerocr.pth'
model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
vocab = vocab
time1 = time.time()


def predict(model, img):
    img = process_input(img, config['dataset']['image_height'],
                        config['dataset']['image_min_width'], config['dataset']['image_max_width'])
    img = img.to(config['device'])
    device = img.device
    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)
        translated_sentence = [[1]*len(img)]
        max_length = 0
        while max_length <= 128 and not all(np.any(np.asarray(translated_sentence).T == 2, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
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
    print(time.time() - time1)
    return s

for i in range(2, 4):
    image = cv2.imread('{}.png'.format(i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    p = threading.Thread(target=predict, args=(model, img))
    p.start()
    p.join()
print('last {}'.format(time.time() - time1))
