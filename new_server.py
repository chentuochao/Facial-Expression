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
#import GUI


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
        self.queue = queue.Queue(maxsize=Config.max_pic_num * 5) # elements in the queue are tuples (img, timestamp)
        print (" New thread started for "+ip+":"+str(port))


    def run(self):
        global is_exit
        while not is_exit:
            length = recvall(self.sock, 16)
            encodedImg = np.frombuffer(recvall(self.sock, int(length)), dtype='uint8')
            decImg = cv2.imdecode(encodedImg, 1)
            time_sample = float(recvall(self.sock,32))
            self.queue.put((time_sample, decImg))
        self.sock.send(b'e')
        print(self.ip + ' Client end capture!')
        while True:
            length = recvall(self.sock, 16)
            if int(length) == 0:
                self.queue.put((0, 0))
                break
            encodedImg = np.frombuffer(recvall(self.sock, int(length)), dtype='uint8')
            decImg = cv2.imdecode(encodedImg, 1)
            time_sample = float(recvall(self.sock,32))
            #print(self.ip, time_sample)
            self.queue.put((time_sample, decImg))        
        
        print('End socket!')


    def getQueue(self):
        return self.queue


def sync_camera(threads, init_time):
    global is_exit
    cnt = 100
    queueList = []
    max_index = -1
    timestampList = np.zeros((num_view,))
    timestampList = timestampList.tolist()
    imgList = np.zeros((num_view, 480, 640, 3))
    for idx, thread in enumerate(threads):
        queueList.insert(idx, thread.getQueue())
    is_end = 0
    while 1:
        #if is_exit: print(str(cnt)+'a')
        for i in range(num_view):
            if i != max_index:
                #print(i, queueList[i].qsize())
                a,b = queueList[i].get()
                if is_exit == 1 and a == 0 and b == 0:
                    is_end = 1
                    break
                timestampList[i], imgList[i] = a,b
        if is_end: break
        #print(timestampList)
        if max(timestampList) - min(timestampList) > 0.02:
            max_index = timestampList.index(max(timestampList))
        else:
            #print('saving'+ str(cnt))
            max_index = -1
            for i in range(num_view):
                cv2.imwrite(save_dir+'/'+str(i)+'/'+str(cnt)+'.jpg', imgList[i])
            cnt += 1
    print('end sync thread')


if __name__ == '__main__':
    save_dir = Config.folder_name[0]
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    TCP_IP = Config.ip_add
    TCP_PORT = Config.ip_port

    total_thread = []

    num_view = Config.view
    BUFFER_SIZE = 1024
    time_interal = 0.1

    if not os.path.exists(save_dir):
            print('creating ./images and subfolders...')
            os.mkdir(save_dir)
            for i in range(num_view):
                os.mkdir(save_dir + '/'+ str(i))
    else:
        if os.path.exists(save_dir + "/collect_timestamp.txt"):
            print('ERROR: collect_timestamp.txt exists.')
            print('\a')
            sys.exit()
        else:
            for i in range(num_view):
                sub_dir_file_names = os.listdir(save_dir + "/" + str(i) + "/")
                images_num = len(sub_dir_file_names)
                if not images_num == 0:
                    print('ERROR: images is not an empty folder.')
                    print('\a')
                    sys.exit()
            print('file and subfolders already exist, but they are empty, continue...')

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

    init_time = time.time()
    thread_sync = Thread(target=sync_camera, args=(threads, init_time))
    thread_sync.setDaemon(True)
    thread_sync.start()
    total_thread.append(thread_sync)
    print('sync begin')

    #GUI.main()
    #is_exit = True

    while 1:
        alive = False
        for i in range(0, len(total_thread)):
            alive = alive or total_thread[i].isAlive()
        if not alive:
            break
    tcpsock.close()
    print('sync end')

