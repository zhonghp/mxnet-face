#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 15:37:10
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-10 17:18:27

import mxnet as mx

ts = 0.1
epoch = 166
face_size = 128
ctx = mx.gpu(0)
# ctx = mx.cpu()
landmarks = 'innerEyesAndBottomLip'
model_prefix = r'../model/lightened_cnn/lightened_cnn'
dlib_model_file = r'../model/dlib/shape_predictor_68_face_landmarks.dat'
