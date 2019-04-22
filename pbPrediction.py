import cv2
import os
import numpy as np
import tensorflow as tf
dir_path = 'E:/Graduation design/Cotton_photos_Record/bestImage/cotton_photos'
# labs = ['gm', 'go', 'lm', 'm', 'sgo', 'slm', 'sm']
preNum = [0, 0, 0, 0, 0, 0, 0]
result = {}
pb_path = "E:/model/test/25947(0.941667)/frozen.pb"

def get_data(images_path, index):
    img = cv2.imread(images_path[index])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = (img/255.0-0.5)*2.0
    return img,images_path[index]


def load_label():
    label =[]
    with open('E:/model/labels.txt', 'r' ,encoding='utf-8') as r:
        lines = r.readlines()
        for l in lines:
            l = l.strip()
            arr = l.split(':')
            label.append(arr[1])
    return label


def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    rightNums = 0
    label = load_label()
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            # input_keep_prob_tensor = sess.graph.get_tensor_by_name("MobilenetV1/Logits/Dropout_1b/dropout/keep_prob:0")
            class_tensor = sess.graph.get_tensor_by_name("classes:0")
            score_tensor = sess.graph.get_tensor_by_name("scores:0")
            # # 定义输出的张量名称
            # output_tensor_name = sess.graph.get_tensor_by_name("MobilenetV1/Logits/SpatialSqueeze:0")
            # 读取测试图片
            for sub_dir in os.listdir(dir_path):
                sub_path = dir_path + '/' + sub_dir
                images_path = [sub_path + '/' + n for n in os.listdir(sub_path)]  # 得到子文件夹的所有图片子根目录
                # 初始化所有类别的值
                preNum = [0, 0, 0, 0, 0, 0, 0]
                for i in range(len(images_path)):
                    im, path = get_data(images_path, i)
                    # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
                    # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
                    # out = sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                    #                                               input_keep_prob_tensor: 1})
                    # # 取概率最大的3个类别及其对应概率
                    # scores = tf.nn.softmax(out, name='pre')
                    # output = tf.nn.top_k(scores, k=3, sorted=True)
                    classes, scores = sess.run([class_tensor, score_tensor], feed_dict={input_image_tensor: im})
                    preNum[classes[0][0]] += 1
                    print('\n识别', path, '结果如下：')
                    for j in range(3):  # top 3
                        idx = classes[0][j]
                        score = scores[0][j]
                        print('\tNo.', j, '类别:', label[idx], '概率:', score)
                rightNum = max(preNum)
                rightNums += rightNum
                result[sub_dir] = {
                    'gm': preNum[0],
                    'go': preNum[1],
                    'lm': preNum[2],
                    'm': preNum[3],
                    'sgo': preNum[4],
                    'slm': preNum[5],
                    'sm': preNum[6],
                    'total_num': len(images_path),
                    'accuracy': float(rightNum/len(images_path))
                }
            for i in result:
                print(i, result[i])
                print("\n")
            print("The accuracy is %f", float(rightNums/5950.0))


if __name__ == '__main__':
    freeze_graph_test(pb_path, dir_path)

