'''
Date: 2022-08-09 11:33:54
LastEditors: OBKoro1
LastEditTime: 2022-08-09 11:41:02
'''
from matplotlib import pyplot as plt
import numpy as np
from skimage import data, color
import cv2

D = 10
img = cv2.imdecode(np.fromfile('./遮盖/遮盖_template_gamma.png',dtype=np.uint8),0)
# numpy 中的傅里叶变换
f1 = np.fft.fft2(img)
f1_shift = np.fft.fftshift(f1)
"""
实现理想高通滤波器 start
"""
s1 = np.log(np.abs(f1_shift))


def ButterworthPassFilter(image, d, n):
    """
    Butterworth 高通滤波器
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (dis / d) ** (2 * n))
        return transform_matrix

    d_matrix = make_transform_matrix(d)
    d_matrix = 1 - d_matrix
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


plt.subplot(221)
plt.axis('off')
plt.title('Original')
plt.imshow(img, cmap='gray')

plt.subplot(222)
plt.axis('off')
plt.title('Butter D=100 n=1')
butter_100_1 = ButterworthPassFilter(img, 100, 1)
plt.imshow(butter_100_1, cmap='gray')

plt.subplot(223)
plt.axis('off')
plt.title('Butter D=30 n=1')
butter_30_1 = ButterworthPassFilter(img, 30, 1)
plt.imshow(butter_30_1, cmap='gray')

plt.subplot(224)
plt.axis('off')
plt.title('Butter D=30 n=5')
butter_30_5 = ButterworthPassFilter(img, 30, 5)
plt.imshow(butter_30_5, cmap='gray')

plt.show()