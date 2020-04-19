import cv2
import dlib
from skimage import io
import os
from pathlib import Path
import pickle
import numpy as np
import random
import Config
import copy
import preprocess
import sys

train_data = []
mode = None

def calibration(q1, q2, q3, p1, p2, p3):
        a = []
        b = []

        p=np.array([ [ p1[0],p1[1],1 ],[ p2[0], p2[1],1 ],[ p3[0], p3[1],1 ] ])
        qx=np.array( [ [q1[0]], [q2[0]], [q3[0]] ])
        qy=np.array( [ [q1[1]], [q2[1]], [q3[1]] ])

        try:
                p_inv = np.linalg.inv(p)
                a = np.matmul(p_inv, qx)
                b = np.matmul(p_inv, qy)
                #print(a, b)
                return a, b
        except np.linalg.LinAlgError as e:
                print(e)
                return False

def convert_position(position ,a, b):    # convert the coordinate of camera to the coordinate of projection 

        """
        param: position shape: n*2
        return: points shape: n*2
        """
        p = np.array([float(position[0]), float(position[1]), 1.])
        points = [np.matmul(p, a), np.matmul(p, b)]
        return points



def convert0(points, begin):
        zero = points[33*2 : 34*2]
        achor1 = points[39*2 : 40*2]
        achor2 = points[42*2 : 43*2]
        q1,q2,q3 = Config.achor
        a, b = calibration(q1,q2,q3, zero, achor1, achor2)
        
        for i in range(0, 68):   
                points[2*i : 2*i+2]  = convert_position(points[2*i : 2*i+2], a,b)    
        zero = copy.deepcopy(points[33*2 : 34*2]) 
        mouthx = np.mean(points[begin*2 : 68*2 : 2])
        mouthy = np.mean(points[begin*2 + 1 : 68*2 : 2])
        for i in range(begin, 68):    
                points[2*i + 0] = points[2*i + 0] - mouthx
                points[2*i + 1] = points[2*i + 1] - mouthy
        #print(scale,achor1)
        return np.array(points[begin*2:])#, np.array([mouthx - zero[0], mouthy - zero[1]])))


def convert_half(points, begin):
        zero = points[33*2 : 34*2]
        achor1 = points[39*2 : 40*2]
        achor2 = points[42*2 : 43*2]
        q1,q2,q3 = Config.achor
        a, b = calibration(q1,q2,q3, zero, achor1, achor2)
        
        for i in range(0, 68):   
                points[2*i : 2*i+2]  = convert_position(points[2*i : 2*i+2], a,b)    
        zero = copy.deepcopy(points[33*2 : 34*2]) 
        mouthx = np.mean(points[begin*2 : 68*2 : 2])
        mouthy = np.mean(points[begin*2 + 1 : 68*2 : 2])
        for i in range(begin, 68):    
                points[2*i + 0] = points[2*i + 0] - mouthx
                points[2*i + 1] = points[2*i + 1] - mouthy
        #print(scale,achor1)
        left = np.hstack((points[48*2:52*2],points[57*2:63*2],points[66*2:68*2]))
        right = np.hstack((points[51*2:58*2],points[62*2:67*2]))
        return right, left

def convert(points, begin):
        achor1 = copy.deepcopy(points[48*2 : 49*2])
        achor2 = copy.deepcopy(points[54*2 : 55*2])
        #print(achor1)
        scale = 130/(achor2[0] - achor1[0])
        for i in range(begin, 68):        
                points[2*i + 0] = scale * (points[2*i + 0] - achor1[0])
                points[2*i + 1] = scale * (points[2*i + 1] - achor1[1])
        #print(scale,achor1)
        return points[begin*2:]

def convert_lefteye(points, begin):
        zero = points[33*2 : 34*2]
        achor1 = points[39*2 : 40*2]
        achor2 = points[42*2 : 43*2]
        q1,q2,q3 = Config.achor
        a, b = calibration(q1,q2,q3, zero, achor1, achor2)
        for i in range(0, 68):   
                points[2*i : 2*i+2]  = convert_position(points[2*i : 2*i+2], a,b)    
        browx = np.mean(points[22*2 : 27*2 : 2])
        browy = np.mean(points[22*2 + 1 : 27*2 : 2])
        eyex = np.mean(points[42*2 : 48*2 : 2])
        eyey = np.mean(points[42*2 + 1 : 48*2 : 2])

        for i in range(22, 27):        
                points[2*i + 0] = points[2*i + 0] - browx
                points[2*i + 1] = points[2*i + 1] - browy
        for i in range(42, 48):        
                points[2*i + 0] = points[2*i + 0] - eyex
                points[2*i + 1] = points[2*i + 1] - eyey
        #print(scale,achor1)
        return np.hstack((points[22*2 : 27*2],points[42*2 : 48*2], [eyex - browx, eyey - browy]))

def convert_righteye(points, begin):
        zero = points[33*2 : 34*2]
        achor1 = points[39*2 : 40*2]
        achor2 = points[42*2 : 43*2]
        q1,q2,q3 = Config.achor
        a, b = calibration(q1,q2,q3, zero, achor1, achor2)
        for i in range(0, 68):   
                points[2*i : 2*i+2]  = convert_position(points[2*i : 2*i+2], a,b)    
        browx = np.mean(points[17*2 : 22*2 : 2])
        browy = np.mean(points[17*2 + 1 : 22*2 : 2])
        eyex = np.mean(points[36*2 : 42*2 : 2])
        eyey = np.mean(points[36*2 + 1 : 42*2 : 2])

        for i in range(17, 22):        
                points[2*i + 0] = points[2*i + 0] - browx
                points[2*i + 1] = points[2*i + 1] - browy
        for i in range(36, 42):        
                points[2*i + 0] = points[2*i + 0] - eyex
                points[2*i + 1] = points[2*i + 1] - eyey
        #print(scale,achor1)
        return np.hstack((points[17*2 : 22*2],points[36*2 : 42*2], [eyex - browx, eyey - browy]))

def convert_mouth_eye(points):
        zero = points[33*2 : 34*2]
        achor1 = points[39*2 : 40*2]
        achor2 = points[42*2 : 43*2]
        q1,q2,q3 = Config.achor
        a, b = calibration(q1,q2,q3, zero, achor1, achor2)
        for i in range(0, 68):   
                points[2*i : 2*i+2]  = convert_position(points[2*i : 2*i+2], a,b)  
              
        # ***************************mouth**********************
        mouthx = np.mean(points[48*2 : 68*2 : 2])
        mouthy = np.mean(points[48*2 + 1 : 68*2 : 2])
        for i in range(48, 68):    
                points[2*i + 0] = points[2*i + 0] - mouthx
                points[2*i + 1] = points[2*i + 1] - mouthy

        # ***************************eye************************
        browx = np.mean(points[22*2 : 27*2 : 2])
        browy = np.mean(points[22*2 + 1 : 27*2 : 2])
        eyex = np.mean(points[42*2 : 48*2 : 2])
        eyey = np.mean(points[42*2 + 1 : 48*2 : 2])

        for i in range(22, 27):        
                points[2*i + 0] = points[2*i + 0] - browx
                points[2*i + 1] = points[2*i + 1] - browy
        for i in range(42, 48):        
                points[2*i + 0] = points[2*i + 0] - eyex
                points[2*i + 1] = points[2*i + 1] - eyey
        #print(scale,achor1)
        return np.hstack((points[48*2:], [mouthx - eyex, mouthy - eyey], points[22*2 : 27*2], points[42*2 : 48*2], [browx - eyex, browy - eyey]))

def label_img(mydir, filename, move):
        global train_data, mode
        img = io.imread(mydir + filename)
        dets = detector(img, 1)
        flag = 0
        begin = 48 #36
        points = np.zeros((68 * 2), dtype=np.int16)
        for k, d in enumerate(dets):
                width = d.right() - d.left()
                heigth = d.bottom() - d.top()
                # 利用预测器预测
                shape = predictor(img, d)
                for i in range(0, 68):        
                        points[2*i + 0] = shape.part(i).x 
                        points[2*i + 1] = shape.part(i).y                          
                flag += 1
        points2 = copy.deepcopy(points)
        if flag != 1:
                print("Invalid Picture and discard it!" + filename+ '  error:' + str(flag))
        else:
                move += [(filename, points2)] 
                if mode == 'mouth': points = convert0(points, begin)
                elif mode == 'righteye': points = convert_righteye(points, begin)
                elif mode == 'lefteye': points = convert_lefteye(points, begin)
                elif mode == 'half': p1, p2 = convert_half(points, begin)
                elif mode == 'both': points = convert_mouth_eye(points)
                else: 
                        print('Invalid mode')
                        raise KeyboardInterrupt 
                #print(points)
                train_data += [(filename, points)]

def visual(mydir, index, move, video):
        pic_name, points = train_data[index]
        #points = np.hstack([right, left])
        pic_name2, points2 = move[index]
        img = io.imread(mydir + pic_name)
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
        #cv2.imshow('face0', img)
        #win = dlib.image_window()
        #win.clear_overlay()
        #win.set_image(img)

        mouth = [0,0]
        #cv2.circle(img, (int( mouth[0] + 100) , int(mouth[1] + 50 )), 1, (0, 0, 255), -1, 8)
        for i in range(points.shape[0]//2):
                #print(points[i][0], type(points[i][0]))
                cv2.circle(img, (int(points[2*i+0] + mouth[0] + 100) , int(points[2*i+1] + mouth[1] + 50 )), 1, (0, 255, 0), -1, 8)
        for i in range(points2.shape[0]//2):
                #print(i, points2[2*i+0] , points2[2*i+1])
                cv2.circle(img, (points2[2*i+0] , points2[2*i+1] ), 1, (0, 0, 255), -1, 8)
       
        # 显示一下处理的图片，然后销毁窗口
        cv2.putText(img, str(index + 102),(50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        video.write(img)
        #cv2.imshow('face', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def visual_eye_mouth(mydir, index, move, video):
        pic_name, points = train_data[index]
        pic_name2, points2 = move[index]
        img = io.imread(mydir + pic_name)

        eye = [100, 50]
        brow = [eye[0] + points[-2], eye[1] + points[-1]]
        mouth = [eye[0] + points[40], eye[1] + points[41]]
        cv2.circle(img, (int(eye[0]), int(eye[1])), 1, (0, 0, 255), -1, 8)
        for i in range(0, 20): # draw mouth
                cv2.circle(img, (int(points[2*i+0] + mouth[0]) , int(points[2*i+1] + mouth[-1] )), 1, (0, 255, 0), -1, 8)
        for i in range(21, 26): # draw brow
                cv2.circle(img, (int(points[2*i+0] + brow[0]) , int(points[2*i+1] + brow[1] )), 1, (0, 255, 0), -1, 8)
        for i in range(26, 32):
                cv2.circle(img, (int(points[2*i+0] + eye[0]) , int(points[2*i+1] + eye[1])), 1, (0, 255, 0), -1, 8)
          
        for i in range(points2.shape[0]//2):
                #print(points[i][0], type(points[i][0]))
                cv2.circle(img, (points2[2*i+0] , points2[2*i+1] ), 1, (0, 0, 255), -1, 8)
        # 显示一下处理的图片，然后销毁窗口
        cv2.putText(img, str(index + 102),(50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        video.write(img)   
        #cv2.imshow('face', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()    

def visual_eye(mydir, index, move, video):
        pic_name, points = train_data[index]
        pic_name2, points2 = move[index]
        img = io.imread(mydir + pic_name)

        mouth = points[-2:]
        cv2.circle(img, (int( mouth[0] + 100) , int(mouth[1] + 50 )), 1, (0, 0, 255), -1, 8)
        for i in range(0, 5):
                cv2.circle(img, (int(points[2*i+0] + 100) , int(points[2*i+1] + 50 )), 1, (0, 255, 0), -1, 8)
        for i in range(5, points.shape[0]//2 - 1):
                cv2.circle(img, (int(points[2*i+0] + mouth[0] + 100) , int(points[2*i+1] + mouth[1] + 50 )), 1, (0, 255, 0), -1, 8)
          
        for i in range(points2.shape[0]//2):
                #print(points[i][0], type(points[i][0]))
                cv2.circle(img, (points2[2*i+0] , points2[2*i+1] ), 1, (0, 0, 255), -1, 8)
        # 显示一下处理的图片，然后销毁窗口
        cv2.putText(img, str(index + 102), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        video.write(img)


def label_folder(folder, down , up):
        global mode 
        global train_data
        train_data = []
        mydir = folder + '/0/'
        mydir0 = folder
        my_dir = Path(mydir) 
        move = []
        if my_dir.exists():
                print('Begin labeling!')
                filelist = os.listdir(my_dir)
                '''
                for filename in filelist:
                        name = filename.split('.', 1)
                        if int(name[0]) < down or  int(name[0]) > up : continue
                '''
                for i in range(down, up):
                        filename = str(i) + '.jpg' 
                        print('Begin labeling Pic: ' + filename)
                        label_img(mydir, filename, move)
                fps = 16 #视频每秒24帧
                size = (640, 480) #需要转为视频的图片的尺寸
                #可以使用cv2.resize()进行修改
                video = cv2.VideoWriter(folder + "/result.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
                for index in range(len(train_data)):
                        if mode == 'mouth' or mode == 'half': visual(mydir, index, move, video)
                        elif mode=='righteye' or mode == 'lefteye': visual_eye(mydir, index, move, video)
                        elif mode == "both": visual_eye_mouth(mydir, index, move, video)
                video.release()
                pickle.dump(train_data, open(mydir0 + "/sliced_data.pkl", "wb"))

if __name__ == '__main__':  # 枚举path路劲下的所有wav和txt文件
        # 使用特征提取器get_frontal_face_detector
        mode = sys.argv[1]
        
        detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        for name in Config.folder_name:
                bound = np.loadtxt(name + '/bound.txt',dtype=int,comments='#')
                #label_folder(name, bound[0], bound[1])
                preprocess.main(name)
                
                