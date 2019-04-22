import tensorflow as tf
from nets.MobileNetV1 import mobilenet_v1, mobilenet_v1_arg_scope
import numpy as np

slim = tf.contrib.slim
CKPT = 'E:/model/test/25947(0.941667)/model.ckpt-25947'


def build_model(inputs):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        logits, end_points = mobilenet_v1(inputs, is_training=False, depth_multiplier=1.0, num_classes=7)
    scores = end_points['Predictions']
    print(scores)
    # 取概率最大的3个类别及其对应概率
    output = tf.nn.top_k(scores, k=3, sorted=True)
    # indices为类别索引，values为概率值
    return output.indices, output.values


def load_model(sess):
    loader = tf.train.Saver()
    loader.restore(sess, CKPT)


inputs = tf.placeholder(dtype=tf.float32, shape=(1, 224, 224, 3), name='input')
classes_tf, scores_tf = build_model(inputs)
classes = tf.identity(classes_tf, name='classes')
scores = tf.identity(scores_tf, name='scores')
with tf.Session() as sess:
    load_model(sess)
    graph = tf.get_default_graph()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [classes.op.name, scores.op.name])
    tf.train.write_graph(output_graph_def, 'E:/model/test/15576', 'frozen.pb', as_text=False)
