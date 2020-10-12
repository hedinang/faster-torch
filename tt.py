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
# from torch.multiprocessing import Pool, Process, set_start_method
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
device = 'cpu'

model, vocab = build_model(config)
weights = 'transformerocr.pth'

# if config['weights'].startswith('http'):
#     weights = download_weights(config['weights'])
# else:
#     weights = config['weights']

model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
# model.to(torch.device('cpu'))
# model.share_memory()
cnn = model.cnn
transformer = model.transformer
cnn.share_memory()
transformer.share_memory()
# model.eval()
vocab = vocab
time1 = time.time()


# def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
#     "data: BxCXHxW"
#     model.eval()
#     device = img.device

#     with torch.no_grad():
#         src = model.cnn(img)
#         memory = model.transformer.forward_encoder(src)

#         translated_sentence = [[sos_token]*len(img)]
#         max_length = 0

#         while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):

#             tgt_inp = torch.LongTensor(translated_sentence).to(device)

# #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
# #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
#             output = model.transformer.forward_decoder(tgt_inp, memory)
#             output = output.to('cpu')

#             values, indices = torch.topk(output, 5)

#             indices = indices[:, -1, 0]
#             indices = indices.tolist()

#             translated_sentence.append(indices)
#             max_length += 1

#             del output

#         translated_sentence = np.asarray(translated_sentence).T

#     return translated_sentence


def predict(cnn,transformer, img):
    img = process_input(img, config['dataset']['image_height'],
                        config['dataset']['image_min_width'], config['dataset']['image_max_width'])
    img = img.to(config['device'])
    device = img.device

    with torch.no_grad():
        src = cnn(img)
        memory = transformer.forward_encoder(src)

        translated_sentence = [[1]*len(img)]
        max_length = 0

        while max_length <= 128 and not all(np.any(np.asarray(translated_sentence).T == 2, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)

#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output = transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T




    # s = translate(img, model)[0].tolist()
    s = translated_sentence[0].tolist()
    s = vocab.decode(s)
    print(time.time() - time1)
    return s

# image = cv2.imread('{}.png'.format(0))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img = Image.fromarray(image)
# predict(cnn,transformer, img)
for i in range(3):

    image = cv2.imread('{}.png'.format(i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    p = threading.Thread(target=predict, args=(cnn,transformer, img))
    p.start()
    p.join()
print('last {}'.format(time.time() - time1))
