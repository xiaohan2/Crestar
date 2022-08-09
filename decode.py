'''
Date: 2022-08-09 14:33:40
LastEditors: OBKoro1
LastEditTime: 2022-08-09 15:24:41
'''
# -*- coding:utf-8 -*-
"""
@author: HinGwenWoong
@license: MIT
"""
import datetime
import time
from pathlib import Path
import numpy as np
import cv2
from pyzbar import pyzbar


def get_qrcode_result(image_input, binary_max=230, binary_step=2):
    """
    获取二维码的结果
    :param image_input: 输入图片数据
    :param binary_max: 二值化的最大值
    :param binary_step: 每次递增的二值化步长
    :return: pyzbar 预测的结果
    """
    # 把输入图像灰度化
    if len(image_input.shape) >= 3:
        image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)

    # 获取自适配阈值
    binary, _ = cv2.threshold(image_input, 0, 255, cv2.THRESH_OTSU)

    # 二值化递增检测
    res = []
    while (binary < binary_max) and (len(res) == 0):
        binary, mat = cv2.threshold(image, binary, 255, cv2.THRESH_BINARY)
        res = pyzbar.decode(mat)
        binary += binary_step

    return res


if __name__ == '__main__':

    image_dir = Path(r"./划痕")
    total_image = 0
    success_count = 0
    for image_file in Path.iterdir(image_dir):
        if image_file.suffix not in [".jpg", ".png"]:
            # 非图片，跳过该文件
            continue

        # 使用 cv2.imdecode 可以读取中文路径下的图片
        image = cv2.imdecode(np.fromfile(image_file, 
                                         dtype=np.uint8), 
                             0)

        start_time = time.time()
        result = get_qrcode_result(image, binary_max=230, binary_step=2)
        
        
        # print(f"Got {image_file} result: {result}, "
        #       f"using time : {datetime.timedelta(seconds=(time.time() - start_time))}")

        if len(result) > 0:
            success_count += 1
        total_image += 1
        if len(result):
            left = result[0].rect.left
            top = result[0].rect.top
            h = result[0].rect.height
            w = result[0].rect.width
            roi = image[top:top+h,left:left+w]
            cv2.imwrite('./qrcode_crop/scratch/'+str(total_image-1)+'.png',roi)
    print(f"total image = {total_image}, success count = {success_count}")

