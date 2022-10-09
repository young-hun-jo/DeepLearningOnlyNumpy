# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero

from PIL import Image
from dezero.models import VGG16
from dezero.datasets import ImageNet

# load input image
url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/blob/images/zebra.jpg?raw=true'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.preprocess(img)
x = x[np.newaxis]

# model
model= VGG16(pretrained=True)

with dezero.test_mode():
    y = model(x)
pred_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels = ImageNet.labels()
print('## Prediction result:', labels[pred_id])







