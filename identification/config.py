#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 15:37:10
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 22:44:43

ts = 0.1
epoch = 166
channel_num = 3 # 1  # set 3 when using vgg-face
face_size = 224 # 128  # set 224 when using vgg-face
gpu_id = 0  # set -1 to use cpu mode
batch_size = 1
feature_size = 4096 # 256  # set 4096 when using vgg-face
exts = [".jpg", ".png"]

extractor = 'vgg_face' # 'lightened_cnn'  # set vgg_face when using vgg-face

landmarks = 'innerEyesAndBottomLip'
model_prefix = r'../model/lightened_cnn/lightened_cnn_gpu'
dlib_model_file = r'../model/dlib/shape_predictor_68_face_landmarks.dat'

mean_val = [129.1863, 104.7624, 93.5940]  # mean value (b, g, r)
vgg_model_def = r'../model/vgg_face_caffe/VGG_FACE_deploy.prototxt'
vgg_model_weights = r'../model/vgg_face_caffe/VGG_FACE.caffemodel'
