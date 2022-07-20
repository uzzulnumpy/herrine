# MNIST

> 머신러닝계의 "Hello World"



## 1. 개요

- 0부터 9까지의 손글씨 이미지로 구성
- 훈련 데이터가 6만장, 테스트 데이터가 1만장
- 각 데이터는 이미지와 라벨로 이루어짐
- 각 이미지는 28X28 해상도의 흑백사진
- 각 픽셀은 0에서 255로 밝기 표현



## 2. 이미지와 텐서

> tensor란?
>
> https://rekt77.tistory.com/102

: 데이터의 배열. 텐서의 랭크는 몇 차원 배열인가를 의미한다.



흑백 이미지: 행렬

흑백 이미지들의 데이터: 3차원 텐서

컬러 이미지: 3차원 텐서(RGB)

컬러 이미지들의 데이터: 4차원 텐서



## 3. PIL 라이브러리

: Python에서 이미지를 처리해주는 라이브러리





## 4. flatten - reshape

flatten으로 텐서를 펼치고, reshape으로 펼쳐진 텐서 행렬을 이미지로 만든다.

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
```

