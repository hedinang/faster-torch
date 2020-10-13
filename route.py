from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from ocr import Ocr
from detect import Detect
import time
import pytesseract
import cv2
from pdf2image import convert_from_path
from datetime import datetime

app = Flask(__name__)
ocr = Ocr()
detect = Detect()


@app.route('/hello', methods=['GET'])
def get():
    return 'hello i am dung'


@app.route('/process', methods=['POST'])
def processUrl():

    log = open("log.txt", "a")
    result = []
    if request.method == 'POST':
        for e in request.files:
            try:
                f = request.files[e]
                timestamp = datetime.now().timestamp()
                name_pdf = '{}.pdf'.format(timestamp)
                name_img = '{}.png'.format(timestamp)
                f.save(secure_filename(name_pdf))

                pages = convert_from_path(
                    name_pdf, 500, size=1200)
                first_page = pages[0]
                first_page.save(name_img, 'png')
                im = cv2.imread(name_img)
                rs = detect.forward(im)
                if rs['secrete'] != None:
                    return {'secrete': True}

                ocr.forward(im, rs)
                # result.append(ocr.process())
                log.write('{}   {}\n'.format(f.filename, 'successful'))
            except Exception as identifier:
                log.write('{}   {}\n'.format(f.filename, 'failed'))
                print(identifier)
            finally:
                os.remove(name_img)
                os.remove(name_pdf)
        log.close()

    return {'result': result}


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
