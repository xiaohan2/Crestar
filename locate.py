'''
Date: 2022-08-10 10:08:36
LastEditors: OBKoro1
LastEditTime: 2022-08-10 16:44:54
'''
import cv2
import numpy as np
import math
def detect(image):
    '''提取所有轮廓'''
    canny = cv2.Canny(image, 100, 200,3)#Apply Canny edge detection on the gray image
    contours,hierachy=cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#Find contours with hierarchy
    return contours,hierachy

def conter(image):
    # 输入包含二维码的图片，返回二维码的实际位置以及朝向
    #图像的宽高和实际的距离
    contours,hierarchy=detect(image)
    centers = {}
    M = {}
    """
    Calculate the centers of the contours
    :param contours: Contours detected with find_contours
    :return: object centers as numpy array
    """

    for i in range(len(contours)):
        M[i] = cv2.moments(contours[i])
        if(M[i]["m00"] == 0):
            centers[i] = (float("nan"), float("nan"))
        else:
            centers[i] = (float(M[i]["m10"] / M[i]["m00"]), float(M[i]["m01"] / M[i]["m00"]))

    hierarchy = hierarchy[0]
    tempindex1 = 0
    tempindex2 = 0
    candicate_contours = []
    for i in range(len(contours)):
        if hierarchy[i][2]==-1:
            continue
        else:
            tempindex1 = hierarchy[i][2] 
        if hierarchy[tempindex1][2]==-1:
            continue
        else:
            tempindex2 = hierarchy[tempindex1][2]
            candicate_contour = [i,tempindex1,tempindex2]
            candicate_contours.append(candicate_contour)
    for contour in candicate_contours:
        parent_contour = contours[contour[0]]
        child_contour1 = contours[contour[1]]
        length1 = cv2.arcLength(parent_contour,1)
        length2 = cv2.arcLength(child_contour1,1)
        if abs(length1/length2-2) > 1:
            candicate_contours.remove(contour)
        else:
            cv2.drawContours(image,contours,contour[0],(0,0,255),2)
  #计算符合5层嵌套的回形框
    mark = 0
    # hierarchy = hierarchy[0]
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c+1
        if hierarchy[k][2] != -1:
            c = c+1
        if c >= 5:
            if mark == 0 :A = i
            elif mark == 1 :B = i
            elif mark == 2 :C = i
            mark = mark + 1
    #给返回值赋初值，如果没有识别到三个回形框，返回-1
    RotationAngle = -1
    if mark >=3:
        #计算三个回形框的质心点之间的距离
        AB = cv_distance(centers[A],centers[B])
        BC = cv_distance(centers[B],centers[C])
        CA = cv_distance(centers[C],centers[A])
        #最长的斜边是right点和bottom点的连线，另一个点即为top点
        if AB > BC and AB > CA:
            outlier = C
            median1 = A
            median2 = B
        elif CA > AB and CA > BC:
            outlier = B
            median1 = A
            median2 = C
        elif BC > AB and BC > CA:
            outlier = A
            median1 = B
            median2 = C
        top = outlier
        
        #斜边的的中点就是二维码的质心
        CentralPoint_x = (centers[median1][0]+centers[median2][0])/2
        CentralPoint_y = (centers[median1][1]+centers[median2][1])/2
        CentralPoint = [CentralPoint_x,CentralPoint_y]

		# //判断二维码旋转方向，通过求top点在对角线哪一侧
		# // 定义：平面上的三点A(x1,y1),B(x2,y2),C(x3,y3)的面积量：
		# // S(A,B,C)=|A B C|= (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3)
		# // 令矢量的起点为A，终点为B，判断的点为C， 
		# // 如果S（A，B，C）为正数，则C在矢量AB的左侧； 
		# // 如果S（A，B，C）为负数，则C在矢量AB的右侧； 
		# // 如果S（A，B，C）为0，则C在直线AB上
        # 	//定义一个位于二维码左上方各200个像素的的DefaultTopPoint，#这样算出来的0度方向是水平向右
        # //当二维码旋转0度时，top点一定在此DefaultTopPoint与二维码质心CentralPoint的连线上
      
        DefaultTopPoint_x = CentralPoint_x + 200
        DefaultTopPoint_y = CentralPoint_y - 200
        DefaultTopPoint = [DefaultTopPoint_x,DefaultTopPoint_y]
        #沿逆时针方向增大到359
        Sdirection = (DefaultTopPoint_x - centers[top][0]) * (CentralPoint_y - centers[top][1]) -  (DefaultTopPoint_y - centers[top][1]) * (CentralPoint_x - centers[top][0])
        
        if Sdirection == 0:
            if centers[top][0]<CentralPoint_x: RotationAngle = 0#关键点的x坐标小于旋转中心的x坐标
            else:RotationAngle = 180
        else: 
            # //通过余弦定理，已知三边求角度
            aa = cv_distance(DefaultTopPoint,centers[top])
            bb = cv_distance(centers[top],CentralPoint)
            cc = cv_distance(CentralPoint,DefaultTopPoint)
            RotationAngle =  math.degrees(math.acos((aa*aa-bb*bb-cc*cc)/(-2*bb*cc)))#旋转角
            if Sdirection < 0: RotationAngle = 360-RotationAngle
        # print(centers[top],' ',centers[median1],' ',centers[median2])
        # cv2.drawContours(image,contours,top,(0,0,255),2)
        # cv2.drawContours(image,contours,median1,(0,0,255),2)
        # cv2.drawContours(image,contours,median2,(0,0,255),2)
    cv2.imshow('hg',image)
    cv2.waitKey()
    return RotationAngle

def Conter(local_x,local_y,image,image_width,image_high,map_width,map_high):
    contours,hierarchy=detect(image)
    centers = {}
    M = {}
    for i in range(len(contours)):
        M[i] = cv2.moments(contours[i])
        if(M[i]["m00"] == 0):
            centers[i] = (float("nan"), float("nan"))
        else:
            centers[i] = (float(M[i]["m10"] / M[i]["m00"]), float(M[i]["m01"] / M[i]["m00"]))

    mark = 0
    hierarchy = hierarchy[0]

    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c+1
        if hierarchy[k][2] != -1:
            c = c+1
        if c >= 5:
            if mark == 0 :A = i
            elif mark == 1 :B = i
            elif mark == 2 :C = i
            mark = mark + 1
    realPosition = [-1,-1]
    RotationAngle = -1
    if mark >=3:
        AB = cv_distance(centers[A],centers[B])
        BC = cv_distance(centers[B],centers[C])
        CA = cv_distance(centers[C],centers[A])
        if AB > BC and AB > CA:
            outlier = C
            median1 = A
            median2 = B
        elif CA > AB and CA > BC:
            outlier = B
            median1 = A
            median2 = C
        elif BC > AB and BC > CA:
            outlier = A
            median1 = B
            median2 = C

        top = outlier  
        CentralPoint_x = (centers[median1][0]+centers[median2][0])/2
        CentralPoint_y = (centers[median1][1]+centers[median2][1])/2
        CentralPoint = [CentralPoint_x,CentralPoint_y]

        realPosition_x =  (CentralPoint_x+local_x)/ image_width * map_width
        realPosition_y =  (CentralPoint_y+local_y) / image_high * map_high
        realPosition = [realPosition_x,realPosition_y]

        # 	//定义一个位于二维码左上方各200个像素的的DefaultTopPoint，//
        #（+200，-200），默认朝右是零度。。（-200，-200）默认朝上是零度
        # //当二维码旋转0度时，top点一定在此DefaultTopPoint与二维码质心CentralPoint的连线上
        DefaultTopPoint_x = CentralPoint_x + 200
        DefaultTopPoint_y = CentralPoint_y - 200
        DefaultTopPoint = [DefaultTopPoint_x,DefaultTopPoint_y]

        Sdirection = (DefaultTopPoint_x - centers[top][0]) * (CentralPoint_y - centers[top][1]) -  (DefaultTopPoint_y - centers[top][1]) * (CentralPoint_x - centers[top][0])
        
        if Sdirection == 0:
            if centers[top][0]<CentralPoint_x: RotationAngle = 0#关键点的x坐标小于旋转中心的x坐标
            else:RotationAngle = 180
        else: 
            # //通过余弦定理，已知三边求角度
            aa = cv_distance(DefaultTopPoint,centers[top])
            bb = cv_distance(centers[top],CentralPoint)
            cc = cv_distance(CentralPoint,DefaultTopPoint)
            RotationAngle =  math.degrees(math.acos((aa*aa-bb*bb-cc*cc)/(-2*bb*cc)))#旋转角
            if Sdirection < 0: RotationAngle = 360-RotationAngle
    return realPosition,RotationAngle

def cv_distance(a,b):
    #求出两点的截距
   selfx=a[0]-b[0]
   selfy=a[1]-b[1]
   selflen= math.sqrt((selfx**2)+(selfy**2))
   return selflen

if __name__ == '__main__':
    image = cv2.imdecode(np.fromfile('dirty/20210324111457456_gamma.png',dtype=np.uint8),0)
    RotationAngle = conter(image)
    print(RotationAngle)

