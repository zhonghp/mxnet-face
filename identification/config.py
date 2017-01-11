#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 15:37:10
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 00:35:00

ts = 0.1
epoch = 166
channel_num = 3 # set 3 when using vgg-face
face_size = 224 # 128 # set 224 when using vgg-face
gpu_id = -1 # set -1 to use cpu mode
# ctx = mx.gpu(0)
# ctx = mx.cpu()
landmarks = 'innerEyesAndBottomLip'
model_prefix = r'../model/lightened_cnn/lightened_cnn'
dlib_model_file = r'../model/dlib/shape_predictor_68_face_landmarks.dat'

batch_size = 1
mean_val = [129.1863, 104.7624, 93.5940] # mean value (b, g, r)
vgg_model_def = r'../model/vgg_face_caffe/VGG_FACE_deploy.prototxt'
vgg_model_weights = r'../model/vgg_face_caffe/VGG_FACE.caffemodel'
