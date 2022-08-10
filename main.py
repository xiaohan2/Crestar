import cv2
import os
from cv2 import FILE_STORAGE_FORMAT_JSON
import numpy as np
import math

def selectMax(window, gray, kp):
    kp = list(kp)
    r = window / 2
    a = 0
    if window != 0:
        for i in range(np.array(kp).shape[0]):
            for j in range(i+1, np.array(kp).shape[0]):
                if np.abs(kp[i].pt[0]-kp[j].pt[0]) + np.abs(kp[i].pt[1] - kp[j].pt[1]) <= 2*r:
                    if kp[i].response < kp[j].response:
                        kp.pop(i)                      
                        a = 1
                        break
                    else:
                        kp.pop(j)
                        a = 1
                        break
        if a != 0:
            kp = selectMax(window, gray, kp)

    return kp

def cosine(des1,des2):
    return des1.dot(des2) / np.linalg.norm(des1) * np.linalg.norm(des2)

def getFiles(root):
    fileNames = os.listdir(root)
    filePaths = []
    for fileName in fileNames:
        if fileName.endswith('.png'):
            filePaths.append(root + fileName)
        else:
            continue
    return filePaths

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv2.imshow("Match Result", outimage)
    cv2.waitKey(0)

"""
使用基于ORB的特征匹配方法
"""
def ORBMatch(img1,img2):
    img1 = cv2.medianBlur(img1,3)
    img2 = cv2.medianBlur(img2,3)
    _,img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #init orb
    orb = cv2.ORB_create()
    #find keypoint
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # #非极大值抑制
    # kp1 = selectMax(15,img1,kp1)
    # kp2 = selectMax(15,img2,kp2)
    #find descriptor
    # beblid = cv2.xfeatures2d.BEBLID_create(0.8)
    # kp1, des1 = beblid.compute(img1, kp1)
    # kp2, des2 = beblid.compute(img2, kp2)

    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    #matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    # #应用比率测试
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.8*n.distance:
    #         good.append(m)

    # matches = good
    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance        
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(0.5*(max_distance + min_distance), 20):
            good_match.append(x)

    cos_sim_sum = 0
    angle_match = []
    for x in good_match:
        cos_sim = cosine(des1[x.queryIdx],des2[x.trainIdx])
        cos_sim_sum+=cos_sim
    cos_avg = cos_sim_sum/len(matches)
    for x in good_match:
        if cosine(des1[x.queryIdx],des2[x.trainIdx]) > cos_avg:
            angle_match.append(x)
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
    # # 计算变换矩阵和MASK
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    # wrap = cv2.warpPerspective(img1, M, (img1.shape[1] + img1.shape[1], img1.shape[0] + img1.shape[0]))

    #画出匹配点
    # draw_match(img1, img2, kp1, kp2, good_match)
    match_rate = len(angle_match)/len(matches)
    return match_rate

def compare_img_hist(img1, img2):
    """
     Compare the similarity of two pictures using histogram(直方图)
         Attention: this is a comparision of similarity, using histogram to calculate
 
         For example:
          1. img1 and img2 are both 720P .PNG file,
             and if compare with img1, img2 only add a black dot(about 9*9px),
             the result will be 0.999999999953
 
     :param img1: img1 in MAT format(img1 = cv2.imread(image1))
     :param img2: img2 in MAT format(img2 = cv2.imread(image2))
     :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)
 
    img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)
 
    similarity = cv2.compareHist(img1_hist, img2_hist, 0)
 
    return similarity

def compare_img_p_hash(img1, img2):
    """
     Get the similarity of two pictures via pHash
         Generally, when:
             ham_dist == 0 -> particularly like
             ham_dist < 5  -> very like
             ham_dist > 10 -> different image
 
         Attention: this is not accurate compare_img_hist() method, so use hist() method to auxiliary comparision.
             This method is always used for graphical search applications, such as Google Image(Use photo to search photo)
 
     :param img1:
     :param img2:
     :return:
    """
    hash_img1 = get_img_p_hash(img1)
    hash_img2 = get_img_p_hash(img2)
 
    return ham_dist(hash_img1, hash_img2)

def get_img_p_hash(img):
    """
     Get the pHash value of the image, pHash : Perceptual hash algorithm(感知哈希算法)
 
     :param img: img in MAT format(img = cv2.imread(image))
     :return: pHash value
    """
    hash_len = 32
     # Resize image, use the different way to get the best result
    resize_gray_img = cv2.resize(img, (hash_len, hash_len), cv2.INTER_AREA)
     # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_LANCZOS4)
     # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_LINEAR)
     # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_NEAREST)
     # resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_CUBIC)
 
     # Change the int of image to float, for better DCT
    h, w = resize_gray_img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = resize_gray_img
 
    # DCT: Discrete cosine transform(离散余弦变换)
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(hash_len, hash_len)
    img_list = vis1.flatten()
 
     # Calculate the avg value
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = []
    for i in img_list:
        if i < avg:
            tmp = '0'
        else:
            tmp = '1'
        avg_list.append(tmp)
 
    # Calculate the hash value
    p_hash_str = ''
    for x in range(0, hash_len * hash_len, 4):
        p_hash_str += '%x' % int(''.join(avg_list[x:x + 4]), 2)
    return p_hash_str

def ham_dist(x, y):
    """
    Get the hamming distance of two values.
         hamming distance(汉明距)
     :param x:
     :param y:
     :return: the hamming distance
    """
    assert len(x) == len(y)
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])

from locate import conter
if __name__ == '__main__':
    # categorys = ['遮盖','划痕','污损']
    categorys = ['cover','scratch','dirty']
    for cat in categorys:
        root = './' + cat + '/'
        imgPaths = getFiles(root)
        target = cv2.imdecode(np.fromfile(imgPaths[-1],dtype=np.uint8),0)
        match_scores = []
        for imgName in imgPaths[:-1]:
            img = cv2.imdecode(np.fromfile(imgName,dtype=np.uint8),0)
            angle = conter(img)
            match_rate = ORBMatch(target,img)
            # match_rate = compare_img_hist(target,img)
            # match_rate = compare_img_p_hash(target,img)
            match_scores.append(match_rate)
            print(match_rate)
        max_rate = max(match_scores)
        print(imgPaths[match_scores.index(max_rate)])
        if cat == 'cover' and match_scores.index(max_rate) == 3:
            print("correct!")
        elif cat == 'scratch' and match_scores.index(max_rate) == 7:
            print("correct!")
        elif cat == 'dirty' and match_scores.index(max_rate) == 8:
            print("correct!")
        else:
            print("error!")
    # path1 = 'D:/projects/Crestar/scratch/20210324112737062_gamma.png'
    # path2 = 'D:/projects/Crestar/scratch/划痕_gamma.png'
    # img1 = cv2.imdecode(np.fromfile(path1,dtype=np.uint8),0)
    # img2 = cv2.imdecode(np.fromfile(path2,dtype=np.uint8),0)
    # outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    # outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv2.imshow("Key Points", outimg3)
    # cv2.waitKey(0)