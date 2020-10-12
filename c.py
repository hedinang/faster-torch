from vietocr.tool.translate import build_model, translate_beam_search, process_input, predict
from vietocr.tool.config import Cfg
import cv2
from PIL import Image
import numpy as np
import torch
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
model, vocab = build_model(config)
weights = 'transformerocr.pth'
device = torch.device('cpu')
# if config['weights'].startswith('http'):
#     weights = download_weights(config['weights'])
# else:
#     weights = config['weights']
model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
sub_img = cv2.imread('5.png')
# cv2.imshow('aa',sub_img)
# cv2.waitKey(0)
img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img.astype(np.uint8))
img = process_input(img, config['dataset']['image_height'],
                    config['dataset']['image_min_width'], config['dataset']['image_max_width'])
img = img.to(config['device'])
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
print(s)