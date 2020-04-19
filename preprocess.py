import cv2
import numpy as np
import Config
from numpy import *;#导入numpy的库函数
from pathlib import Path
import os

def remove_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            pixle1 = hsv[i, j, 0]
            r,g,b = img[i, j]
            if  pixle1 > 100 or (r < 30 and g < 30 and b < 30):
                img[i, j] = [0,0,0]
    return img

def cr_otsu(img):
    """YCrCb颜色空间的Cr分量+Otsu阈值分割"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
 
    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    #cv2.namedWindow("image raw", cv2.WINDOW_NORMAL)
    #cv2.imshow("image raw", img)
    #cv2.namedWindow("image CR", cv2.WINDOW_NORMAL)
    #cv2.imshow("image CR", cr1)
    #cv2.namedWindow("Skin Cr+OTSU", cv2.WINDOW_NORMAL)
    #cv2.imshow("Skin Cr+OTSU", skin)
 
    dst = cv2.bitwise_and(img, img, mask=skin)
    #cv2.namedWindow("seperate", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate", dst)
    return dst

def ellipse_detect(img):
    image = 'aaaaaa'
    skinCrCbHist = np.zeros((256,256), dtype= np.uint8 )
    cv2.ellipse(skinCrCbHist ,(113,155),(23,15),43,0, 360, (255,255,255),-1)
 
    YCRCB = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    (y,cr,cb)= cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x,y)= cr.shape
    for i in range(0,x):
        for j in range(0,y):
            CR= YCRCB[i,j,1]
            CB= YCRCB[i,j,2]
            if skinCrCbHist [CR,CB]>0:
                skin[i,j]= 255
    cv2.namedWindow(image, cv2.WINDOW_NORMAL)
    cv2.imshow(image, img)
    dst = cv2.bitwise_and(img,img,mask= skin)
    cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    cv2.imshow("cutout",dst)

def filter(image):
    black = np.zeros(image.shape, dtype= np.uint8)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)#注意cv.THRESH_BINARY_INV
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print("number of contours:%d" % len(contours))
    #cv2.drawContours(binary, contours, 2, (255, 0, 0), 2)
    #cv2.namedWindow("seperate1", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate1", binary)
    #找到最大区域并填充
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(len(contours)):
        cv2.fillConvexPoly(binary, contours[i], 0)
    #cv2.namedWindow("seperate2", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate2", binary)
    cv2.fillConvexPoly(black, contours[max_idx], (255,255,255))
    #cv2.namedWindow("seperate4", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate4", black)
    #print(image.shape, type(image), image.dtype)
    #print(black.shape, type(black), black.dtype)
    dst = cv2.bitwise_and(image, black)
    #cv2.rectangle(dst, (0,230), (244,244), (0,0,0), -1)
    #cv2.namedWindow("seperate3", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate3", dst)
    return dst

def filter2(image):
    black = np.zeros(image.shape, dtype= np.uint8)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)#注意cv.THRESH_BINARY_INV
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print("number of contours:%d" % len(contours))
    #cv2.drawContours(binary, contours, 2, (255, 0, 0), 2)
    #cv2.namedWindow("seperate1", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate1", binary)
    #找到最大区域并填充
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(len(contours)):
        cv2.fillConvexPoly(binary, contours[i], 0)
    #cv2.namedWindow("seperate2", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate2", binary)
    cv2.fillConvexPoly(black, contours[max_idx], (255,255,255))
    #cv2.namedWindow("seperate4", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate4", black)
    #print(image.shape, type(image), image.dtype)
    #print(black.shape, type(black), black.dtype)
    dst = cv2.bitwise_and(image, black)
    #print(type(black),black.shape)
    #cv2.rectangle(black, (0,0), (640,50), (255,255,255), -1)
    #cv2.namedWindow("seperate3", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate3", dst)
    return black[50:380, 200:620]




def erode_demo(image):
    print(image.shape)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)#注意cv.THRESH_BINARY_INV
    cv2.imshow("binary",binary)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#得到结构元素 cross十字交叉 矩形  结构元素变大 腐蚀加重
    dst=cv2.erode(binary,kernel=kernel)
    cv2.imshow("erode_demo",dst)


def dilate_demo(image):
    print(image.shape)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)#注意cv.THRESH_BINARY_INV
    cv2.imshow("binary",binary)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#得到结构元素 cross十字交叉 矩形  结构元素变大 腐蚀加重
    dst=cv2.dilate(binary,kernel=kernel)
    cv2.imshow("dilate_demo",dst)

#全局阈值
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    cv2.namedWindow('grey', cv2.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
    cv2.imshow('grey', gray)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
    cv2.imshow("binary0", binary)

#局部阈值
def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    cv2.namedWindow("binary1", cv2.WINDOW_NORMAL)
    cv2.imshow("binary1", binary)

#用户自己计算阈值
def custom_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:",mean)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
    cv2.imshow("binary2", binary)

#Canny边缘提取
def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv2.Canny(gray, 50, 150)
    cv2.imshow("Canny Edge", edge_output)
    dst = cv2.bitwise_and(image, image, mask= edge_output)
    cv2.imshow("Color Edge", dst)

def main(folder):
    mydir = folder + '/1/'
    mydir0 = folder
    my_dir = Path(mydir) 
    output1 = '/black2/'
    output2 = '/black2/'
    if not os.path.exists(mydir0 + output1):
        os.mkdir(mydir0  + output1)
    if my_dir.exists():
            print('Begin labeling!')
            filelist = os.listdir(my_dir)
            for filename in filelist:
                if (filename[3] == '.' and int(filename[0:3]) < 102) : continue 
                src = cv2.imread(folder + '/1/' + filename)
                dst = src[ 0 : 220, 0 : 640]
                #dst = remove_red(dst)
                #src = np.uint8(np.clip((src + 60), 0, 255)) 
                #dst = cr_otsu(dst)
                #dst = filter2(dst)
                #cv2.imshow('bbb', dst)
                #dst = cv2.medianBlur(dst,5)
                
                #dst = cv2.blur(dst,(3,5))#模板大小3*5
                #kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#得到结构元素 cross十字交叉 矩形  结构元素变大 腐蚀加重
                #dst=cv2.dilate(dst,kernel=kernel)
                #dst=cv2.dilate(dst,kernel=kernel)
                #dst=cv2.erode(dst,kernel=kernel)
                #dst=cv2.erode(dst,kernel=kernel)
                #dst = cv2.medianBlur(dst,5)
                #cv2.imshow('aaa', dst)
                print(folder + output1 + filename)
                cv2.imwrite(folder + output1 + filename, dst)
                #cv2.waitKey(0)
    
    mydir = folder + '/2/'
    mydir0 = folder
    my_dir = Path(mydir) 
    if not os.path.exists(mydir0+output2):
        os.mkdir(mydir0  + output2)
    if my_dir.exists():
            print('Begin labeling!')
            filelist = os.listdir(my_dir)
            for filename in filelist:
                if (filename[3] == '.' and int(filename[0:3]) < 102) : continue 
                src = cv2.imread(folder + '/2/' + filename)  
                dst = cr_otsu(src)
                dst = filter2(dst)
                #cv2.imshow('sds',dst)
                #cv2.waitkey(0)
                print(folder + output2 + filename)
                
                cv2.imwrite(folder + output2 + filename, dst)

    


