import tensorflow as tf
import os

model_dir = 'E:/tmp/test/buffer'
#model_dir = 'E:/Graduation design/model/inception_dec_2015'
model_name = 'frozen_test_model.pb'
#model_name = 'tensorflow_inception_graph.pb'

def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()#
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
tensor_op_list = [tensor.op for tensor in tf.get_default_graph().as_graph_def().node]
tensor_input_list = [tensor.input for tensor in tf.get_default_graph().as_graph_def().node]
for tensor in tf.get_default_graph().as_graph_def().node:
    print(tensor.name, tensor.input, tensor.op,'\n')
# for tensor_name,tensor_input in tensor_name_list,tensor_input_list:
#     print(tensor_name,tensor_input,'\n')
# for tensor in tf.get_default_graph().as_graph_def().node:
#     print(tensor,'\n')
