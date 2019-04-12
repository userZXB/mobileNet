import os
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


slim = tf.contrib.slim

directory = 'E:/Graduation design/Cotton_photos_Record/cotton_photos/sm'
write_directory = 'E:/Graduation design/Cotton_photos_Record/expan_Image'
height = 224
width = 224


def expand_Image(image, i):
    if i==0:
        newImage = tf.image.random_flip_up_down(image)
    elif i==1:
        newImage = tf.image.random_flip_left_right(image)
    elif i==2:
        newImage = tf.image.transpose_image(image)
    elif i==3:
        newImage = tf.image.random_brightness(image, max_delta=0.2)
    elif i==4:
        newImage = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif i==5:
        newImage = tf.image.random_hue(image, max_delta=0.2)
    else:
        newImage = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    newImage = tf.image.convert_image_dtype(newImage, dtype=tf.uint16)
    encoded_image = tf.image.encode_png(newImage)
    return encoded_image


def store_newImage(i, encoded_image, imageName, subname):
    # 将名字加上编号值进行组装
    # imageName包含后缀，所以应该去掉后缀
    subpath = os.path.join(write_directory, subname)
    name = imageName[:-5]  # 截取合法名字
    new = name + str(i) + '.png'  # 进行名字组装
    with tf.gfile.GFile(os.path.join(subpath, new), "wb") as f:
        print(os.path.join(subpath, new))
        f.write(encoded_image.eval())


# 从文件夹读取文件，然后进行扩充，扩充后的图片进行resize()不需要经过其他处理。
with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵
    # TensorFlow还提供了tf.image.decode_png函数对png格式的图像进行解码
    # 解码之后的结果为一个张量 在使用它的取值之前需要明确调用运行的过程
    # 从文件夹获取子文件夹
    # for subname in os.listdir(directory):
    #     path = os.path.join(directory, subname)  # 进入子文件夹
    #     if os.path.isdir(path):
    #         for imageName in os.listdir(path):
    #             imagePath = os.path.join(path, imageName)  # 得到图片的路径
    #             image_raw_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    #             # 读取的是压缩编码的结果，所以需要解码
    #             img_data = tf.image.decode_png(image_raw_data)
    #             # 将数据类型转化为实数方便对图像处理
    #             image = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    #             # 上下左右各得到一张图片
    #             # 调整亮度，对比度，饱和度，色相各得到一张图片
    #             for i in range(7):
    #                 # 根据序号选择扩充的方式,返回的是编码的数据
    #                 data_raw = expand_Image(image, i)
    #                 # 将扩充得到的图片按照序号进行存储
    #                 store_newImage(i, data_raw, imageName, subname)
    if os.path.isdir(directory):
        for imageName in os.listdir(directory):
            imagePath = os.path.join(directory, imageName)  # 得到图片的路径
            image_raw_data = tf.gfile.FastGFile(imagePath, 'rb').read()
            # 读取的是压缩编码的结果，所以需要解码
            img_data = tf.image.decode_png(image_raw_data)
            # 将数据类型转化为实数方便对图像处理
            image = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            # 上下左右各得到一张图片
            # 调整亮度，对比度，饱和度，色相各得到一张图片
            for i in range(7):
                # 根据序号选择扩充的方式,返回的是编码的数据
                data_raw = expand_Image(image, i)
                # 将扩充得到的图片按照序号进行存储
                store_newImage(i, data_raw, imageName, 'sm')




