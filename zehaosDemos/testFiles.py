import cv2
import os
import numpy as np
dir_path = 'E:/tmp/test/origin'


def get_data(images_path, idx):
    img_path = images_path[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = (img/255.0-0.5)*2.0
    return img_path, img


for sub_dir in os.listdir(dir_path):

    sub_path = dir_path + '/' + sub_dir

    images_path = [sub_path + '/' + n for n in os.listdir(sub_path)]  # 得到子文件夹的所有图片路径
    # 初始化所有类别的值
    preNum = [0, 0, 0, 0, 0, 0, 0]

    # 获取该类别的索引值
    for i in range(len(images_path)):
        path, img = get_data(images_path, i)
        print(path)