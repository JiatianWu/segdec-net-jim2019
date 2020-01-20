from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os
import re
import time
from datetime import datetime

import numpy as np
import pylab as plt

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import inspect_checkpoint as chkp

def freeze_graph():

    MODEL_DIR = "/home/jiatian/project/segdec-net-jim2019/output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/folder_0_lite/"
    MODEL_NAME = "frozen_model.pb"

    if not tf.gfile.Exists(MODEL_DIR): #创建目录
        tf.gfile.MakeDirs(MODEL_DIR)

    model_folder = '/home/jiatian/project/segdec-net-jim2019/output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_00'
    checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    output_node_names = "tower_0//conv5/weights/ExponentialMovingAverage,tower_0/total_loss/avg" #原模型输出操作节点的名字
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())

def show_tensor_name():
    checkpoint_path=os.path.join('/home/jiatian/project/segdec-net-jim2019/output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_0/model.ckpt-6599')
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map=reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name: ', key)

def inspect_tensor_name():
    chkp.print_tensors_in_checkpoint_file(file_name="/home/jiatian/project/segdec-net-jim2019/output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_0/model.ckpt-6599", 
                                         tensor_name=None, # 如果为None,则默认为ckpt里的所有变量
                                        all_tensors=False, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                        all_tensor_names=True) # bool 是否打印所有的tensor的name

def convert_pb_lite():
    saved_model_dir = '/home/jiatian/project/segdec-net-jim2019/output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_00'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open("/home/jiatian/project/segdec-net-jim2019/output/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy/fold_00/converted_model.tflite", "wb").write(tflite_model)

if __name__ == "__main__":
    # freeze_graph()
    convert_pb_lite()
    # show_tensor_name()
    # inspect_tensor_name()