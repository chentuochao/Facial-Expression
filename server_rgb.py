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
        self.queue = queue.PriorityQueue(maxsize=550) # elements in the queue are tuples (img, timestamp)
        print (" New thread started for "+ip+":"+str(port))


    def run(self):
        global is_exit
        while not is_exit:
            length = recvall(self.sock, 16)
            encodedImg = np.frombuffer(recvall(self.sock, int(length)), dtype='uint8')
            decImg = cv2.imdecode(encodedImg, 1)
            decImg = cv2.resize(decImg, (244, 244))
            time_sample = float(recvall(self.sock,32))
            self.queue.put((time_sample, decImg))


    def getQueue(self):
        return self.queue


def sync_camera(threads, init_time):
    global is_exit
    image0_dir = save_dir + "/0/"
    sub_dir_file_names = os.listdir(image0_dir)
    images_num = len(sub_dir_file_names)
    
    cnt = 100
    count_loop = 0
    queueList = []
    timestampList = np.zeros((num_view,))
    timestampList = timestampList.tolist()
    imgList = np.zeros((num_view, 244, 244, 3))
    for idx, thread in enumerate(threads):
        queueList.insert(idx, thread.getQueue())
    
    while not is_exit:
        for i in range(num_view):
            #print('Queue %d size = '%i + str(queueList[i].qsize()))
            timestampList[i], imgList[i] = queueList[i].get()
        #print(max(timestampList), min(timestampList) )
        if max(timestampList) - min(timestampList) > 0.1:
            #print('Leave the max time only')
            max_time_idx = timestampList.index(max(timestampList))
            queueList[max_time_idx].put((timestampList[max_time_idx], imgList[max_time_idx]))
        else:
            print('saving')
            for i in range(num_view):
                cv2.imwrite(save_dir+'/'+str(i)+'/'+str(cnt+images_num)+'.jpg', imgList[i])
                with open(save_dir + "/collect_timestamp.txt","a+") as f:
                    f.write(str(timestampList[i])+' ')
            with open(save_dir + "/collect_timestamp.txt","a+") as f:
                f.write('\n')
            cnt += 1
            if cnt%frames_per_cycle == 0:
                # input('Press enter to continue...')
                for i in range(num_view):
                    queueList[i].queue.clear()
                count_loop = count_loop + 1
                print('New cycle begins: ' + str(count_loop))
    print('end thread')

if __name__ == '__main__':
    save_dir = './imagehhh'
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    assert not os.path.exists(save_dir + "/collect_timestamp.txt")
    TCP_IP = '192.168.0.196'
    TCP_PORT = 6666
    max_sync_single_view_images = 2048
    frames_per_cycle = 500
    total_thread = []
    ARAY_LEN = 768 * 3

    num_view = 2
    BUFFER_SIZE = 1024
    time_interal = 0.1

    if not os.path.exists(save_dir):
            print('creating ./images and subfolders...')
            os.mkdir(save_dir)
            for i in range(num_view):
                os.mkdir(save_dir + f'/{i}')
    else:
        if os.path.exists(save_dir + "/collect_timestamp.txt"):
            print('ERROR: collect_timestamp.txt exists.')
            print('\a')
            sys.exit()
        else:
            for i in range(num_view):
                sub_dir_file_names = os.listdir(save_dir + f"/{i}/")
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
    
    while 1:
        alive = False
        for i in range(0, len(total_thread)):
            alive = alive or total_thread[i].isAlive()
        if not alive:
            break
    tcpsock.close()
    print('sync end')

