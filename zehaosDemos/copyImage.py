import os
import tensorflow as tf

slim = tf.contrib.slim

directory = 'E:/Graduation design/Cotton_photos_Record/folder/cotton_photos'
write_directory = 'E:/Graduation design/Cotton_photos_Record/expan_Image'
with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵
    # TensorFlow还提供了tf.image.decode_png函数对png格式的图像进行解码
    # 解码之后的结果为一个张量 在使用它的取值之前需要明确调用运行的过程
    # 从文件夹获取子文件夹
    for subname in os.listdir(directory):
        path = os.path.join(directory, subname)  # 进入子文件夹
        if os.path.isdir(path):
            for imageName in os.listdir(path):
                imagePath = os.path.join(path, imageName)  # 得到图片的路径
                image_raw_data = tf.gfile.FastGFile(imagePath, 'rb').read()
                data = tf.image.decode_png(image_raw_data)
                name = imageName[:-5]
                new = name + '.png'  # 进行名字组装
                newPath = os.path.join(write_directory, subname)
                encoded_data = tf.image.encode_png(data)
                with tf.gfile.GFile(os.path.join(newPath, new), "wb") as f:
                    print(os.path.join(newPath, new))
                    f.write(encoded_data.eval())