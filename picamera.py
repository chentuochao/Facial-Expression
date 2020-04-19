import cv2
import dlib
from skimage import io
import os
from pathlib import Path
import pickle
import numpy as np
import random
import Config
import argparse
import datetime
import imutils
import time
import signal
import socket
import queue
import struct
from threading import Thread
import Config

detector = dlib.get_frontal_face_detector()
# dlib的68点模型，使用作者训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")




def label_img(img):
        # 特征提取器的实例化
        dets = detector(img, 1)
        #print("Face number", len(dets))
        for k, d in enumerate(dets):
                # 利用预测器预测
                shape = predictor(img, d)
                for i in range(0, 68):                               
                        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (255, 255, 255), -1, 8)
                        #cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255))
        return img

is_exit = False
 
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
    def __init__(self, ip, port, sock):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.sock = sock
        self.queue = queue.PriorityQueue(maxsize=9000) # elements in the queue are tuples (img, timestamp)
        print (" New thread started for "+ip+":"+str(port))


    def run(self):
        global is_exit
        while not is_exit:
            length = recvall(self.sock, 16)
            encodedImg = np.frombuffer(recvall(self.sock, int(length)), dtype='uint8')
            decImg = cv2.imdecode(encodedImg, 1)
            decImg = label_img(decImg)
            #decImg = cv2.resize(decImg, (244, 244))
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
        newthread = ClientThread(ip,port,conn)
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

