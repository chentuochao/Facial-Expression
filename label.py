import cv2
import dlib
from skimage import io
import os
from pathlib import Path


def label_img(img):
        # 生成dlib的图像窗
        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(img)

        # 特征提取器的实例化
        dets = detector(img, 1)
        print("人脸数：", len(dets))

        for k, d in enumerate(dets):
                print("第", k+1, "个人脸d的坐标：",
                        "left:", d.left(),
                        "right:", d.right(),
                        "top:", d.top(),
                        "bottom:", d.bottom())
                width = d.right() - d.left()
                heigth = d.bottom() - d.top()
                print('人脸面积为：',(width*heigth))

                # 利用预测器预测
                shape = predictor(img, d)
                # 标出68个点的位置
                for i in range(68):
                        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 4, (0, 255, 0), -1, 8)
                        cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                # 显示一下处理的图片，然后销毁窗口
                cv2.imshow('face', img)
                cv2.waitKey(0)


if __name__ == '__main__':  # 枚举path路劲下的所有wav和txt文件
                # 使用特征提取器get_frontal_face_detector
        detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        mydir = './imagehhh/0/'
        my_dir = Path(mydir) 
        if my_dir.exists():
                print('Begin labeling!')
                filelist = os.listdir(my_dir)
                for filename in filelist:
                        print('Begin labeling Pic: ' + filename)
                        img = io.imread(mydir + filename)
                        label_img(img)
            
                