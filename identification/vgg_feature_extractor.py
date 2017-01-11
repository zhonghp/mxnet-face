#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 23:30:26
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 17:43:19

import copy
import caffe
import numpy as np

import config
from feature_extractor import FeatureExtractor


class VggFeatureExtractor(FeatureExtractor):

    def __init__(self):
        if config.gpu_id < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_device(config.gpu_id)
            caffe.set_mode_gpu()
        self.__net = caffe.Net(config.vgg_model_def,
                               config.vgg_model_weights, caffe.TEST)
        self.__transformer = caffe.io.Transformer(
            {'data': self.__net.blobs['data'].data.shape})
        self.__transformer.set_transpose('data', (2, 0, 1))
        self.__transformer.set_mean('data', np.array(config.mean_val))
        # self.__transformer.set_raw_scale('data', 255)
        # self.__transformer.set_channel_swap('data', (2, 1, 0)) # swap channel
        # from RGB to BGR
        self.__net.blobs['data'].reshape(
            config.batch_size, config.channel_num, config.face_size, config.face_size)

    def extract_feature(self, img_bgr):
        transformed_img = self.__transformer.preprocess('data', img_bgr)
        self.__net.blobs['data'].data[...] = transformed_img
        output = self.__net.forward()
        feature = self.__net.blobs['fc7'].data[0]
        return feature.copy()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        img1 = 'test1.jpg'
        img2 = 'test2.jpg'
    else:
        img1 = sys.argv[1].strip()
        img2 = sys.argv[2].strip()

    import cv2
    import util
    config.channel_num = 3
    config.face_size = 224
    config.feature_size = 4096
    config.extractor = 'vgg_face'

    extractor = VggFeatureExtractor()
    # img_bgr = cv2.imread('../model/vgg_face_caffe/ak.png')
    # feature = extractor.extract_feature(img_bgr)

    img1_bgr = cv2.imread(img1)
    img2_bgr = cv2.imread(img2)
    # img1_bgr = cv2.imread('../../../data/lfw-align/Shane_Loux/Shane_Loux_0001.png')
    # img2_bgr = cv2.imread('../../../data/lfw-align/Val_Ackerman/Val_Ackerman_0001.png')
    feat1 = extractor.extract_feature(img1_bgr)
    feat2 = extractor.extract_feature(img2_bgr)
    print util.cosine_similarity(feat1, feat2)
