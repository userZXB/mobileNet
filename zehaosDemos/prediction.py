import tensorflow as tf
import cv2
import os
import numpy as np
from nets.MobileNetV1 import mobilenet_v1, mobilenet_v1_arg_scope
slim = tf.contrib.slim
CKPT = 'E:/model/test/15576/model.ckpt-15576'
dir_path = 'E:/Graduation design/Cotton_photos_Record/bestImage/cotton_photos'
labs = ['gm', 'go', 'lm', 'm', 'sgo', 'slm', 'sm']
preNum = [0, 0, 0, 0, 0, 0, 0]
# 我可以手动计算一下准确率，使用未处理的照片
def build_model(inputs):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        logits, end_points = mobilenet_v1(inputs, is_training=False, depth_multiplier=1.0, num_classes=7)
    scores = end_points['Predictions']
    print(scores)
    #取概率最大的3个类别及其对应概率
    output = tf.nn.top_k(scores, k=3, sorted=True)
    #indices为类别索引，values为概率值
    return output.indices, output.values


def load_model(sess):
    loader = tf.train.Saver()
    loader.restore(sess, CKPT)



def load_label():
    label =[]
    with open('E:/model/labels.txt', 'r' ,encoding='utf-8') as r:
        lines = r.readlines()
        for l in lines:
            l = l.strip()
            arr = l.split(':')
            label.append(arr[1])
    return label


def get_data(images_path, idx):
    img_path = images_path[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = (img/255.0-0.5)*2.0
    return img_path, img


inputs = tf.placeholder(dtype=tf.float32, shape=(1, 224, 224, 3))
classes_tf, scores_tf = build_model(inputs)
# 获得所有子文件的路径
sub_paths = []
# 存放图片识别结果,按照字典的形式存放
result = {}
#
# result['gm'] = {
#    'righNum':
# }
label = load_label()
with tf.Session() as sess:
    load_model(sess)
    print("循环验证中")
    for sub_dir in os.listdir(dir_path):
        sub_path = dir_path + '/' + sub_dir
        images_path = [sub_path + '/' + n for n in os.listdir(sub_path)]  # 得到子文件夹的所有图片路径
        # 初始化所有类别的值
        preNum = [0, 0, 0, 0, 0, 0, 0]
        # 获取该类别的索引值
        for i in range(len(images_path)):
            path, img = get_data(images_path, i)
            classes, scores = sess.run([classes_tf, scores_tf], feed_dict={inputs: img})
            # begin = path.index('-')
            # end = path.index('(')
            # ground_truth = path[begin + 1:end]
            preNum[classes[0][0]] += 1
            print('\n识别', path, '结果如下：')
            for j in range(3):  # top 3
                idx = classes[0][j]
                score = scores[0][j]
                print('\tNo.', j, '类别:', label[idx], '概率:', score)
        result[sub_dir] = {
            'gm': preNum[0],
            'go': preNum[1],
            'lm': preNum[2],
            'm' : preNum[3],
            'sgo': preNum[4],
            'slm': preNum[5],
            'sm': preNum[6],
            'total_num': len(images_path)
        }
    for i in result:
        print(i, result[i])
        print("\n")

#
#  扩充后的图片（5960张）各个类别的准确率以及所有类别的平均准确率
#  类别\识别结果|| gm    go   lm    m    sgo   slm    sm   total_num    accuracy
#    gm         || 784     0    0    4     0     0    74    862         784/862=90%
#    go         || 0      499  63   76    5     190   1     834         60%
#    lm         || 1      1    510  196   0     103   32    843         85%
#     m         || 33     0    0    785   0     23    16    857         96%
#    sgo        || 1      9    54   155   457   174    0    850         54%
#    slm        || 10     0    1     98   0     728   21    858         60%
#    sm         || 18     0    0     13   0     5     820   856         92%
# average accuracy 约为77%

# --------------------------------------
# 原始图片（756张）准确率以及所有类别的平均准确率
# 类别\识别结果|| gm    go   lm    m    sgo   slm    sm   total_num    accuracy
#    gm        || 102    0    0    0     0     0    6     108         102/108=94%
#    go        ||  0     87   3    5     0     13   0     108         77%
#    lm        ||  0     0    94   4     0     9    1     108         87%
#     m        ||  3     0    0    104   0     1    0     108         96%
#    sgo       ||  0     0    7    9    81    11    0     108         75%
#    slm       ||  1     0    0    10   0     95    2     108         88%
#    sm        ||  1     0    0    2    0     0    105    108         97%
# average accuracy 约为88%
