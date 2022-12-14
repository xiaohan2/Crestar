'''
Date: 2022-08-09 16:06:45
LastEditors: OBKoro1
LastEditTime: 2022-08-09 16:09:49
'''
import numpy as np
import cv2

image = cv2.imread("./qrcode_crop/cover/0.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
image_blur = cv2.medianBlur(image,3)
cv2.imshow("Original",image)

#Sobel边缘检测
sobelX = cv2.Sobel(image_blur,cv2.CV_64F,1,0)#x方向的梯度
sobelY = cv2.Sobel(image_blur,cv2.CV_64F,0,1)#y方向的梯度

sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值

sobelCombined = cv2.bitwise_or(sobelX,sobelY)#
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey()