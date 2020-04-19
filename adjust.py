r"""Receives images in continuous tracking scenario only if they are aligned.

Inputs:
    none

Outputs:
    received images
    aligned timestamp of images
"""

import signal
import os
import cv2
import sys
import socket
import time
import queue
import struct
import numpy as np
from threading import Thread
import Config

is_exit = False

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
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(len(contours)):
        cv2.fillConvexPoly(binary, contours[i], 0)
    #cv2.namedWindow("seperate2", cv2.WINDOW_NORMAL)
    #cv2.imshow("seperate2", binary)
    cv2.fillConvexPoly(black, contours[max_idx], (255,255,255))
    dst = cv2.bitwise_and(image, black)
    #dst = cv2.bitwise_and(image, black)
    return dst


def handler(signum, frame):
    global is_exit
    is_exit = True
    print("receive a signal %d, is_exit = %d"%(signum, is_exit))

def recvall(socksock, countcount):
    buf = b''
    while countcount:
        newbuf = socksock.recv(countcount)
        if not newbuf: return None
        buf += newbuf
        countcount -= len(newbuf)
    return buf


class ClientThread(Thread):
    def __init__(self, ip, port, sock, mode):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.sock = sock
        self.mode = mode
        self.queue = queue.PriorityQueue(maxsize=9000) # elements in the queue are tuples (img, timestamp)
        print (" New thread started for "+ip+":"+str(port))


    def run(self):
        global is_exit
        while not is_exit:
            length = recvall(self.sock, 16)
            encodedImg = np.frombuffer(recvall(self.sock, int(length)), dtype='uint8')
            decImg = cv2.imdecode(encodedImg, 1)
            if mode == '1':
            #decImg = cv2.resize(decImg, (244, 244))
                dst = cr_otsu(decImg)
                decImg = filter2(dst)
            cv2.imshow("Security Feed", decImg)
            time_sample = float(recvall(self.sock,32))
            amp = 0xFF
            key = cv2.waitKey(1) & amp
            # 如果q键被按下，跳出循环
            if key == ord("q"):
                break
            #self.queue.put((time_sample, decImg))


    def getQueue(self):
        return self.queue


if __name__ == '__main__':
    save_dir = Config.folder_name
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    TCP_IP = Config.ip_add
    TCP_PORT = Config.ip_port
    max_sync_single_view_images = 2048
    frames_per_cycle = 500
    total_thread = []
    ARAY_LEN = 768 * 3

    num_view = 1
    BUFFER_SIZE = 1024
    time_interal = 0.1

    mode = sys.argv[1]

    tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpsock.bind((TCP_IP, TCP_PORT))
    threads = []

    tcpsock.listen(num_view)
    print("please open Raspberry Pi's with the order: 100->139->161->198")

    for i in range(num_view):
        print ("Waiting for incoming connections...")
        (conn, (ip,port)) = tcpsock.accept()
        print ('Got connection from ', (ip,port))
        newthread = ClientThread(ip,port,conn,mode)
        newthread.setDaemon(True)
        newthread.start()
        threads.append(newthread)
        total_thread.append(newthread)
    
    while 1:
        alive = False
        for i in range(0, len(total_thread)):
            alive = alive or total_thread[i].isAlive()
        if not alive:
            break
    tcpsock.close()
    print('sync end')

