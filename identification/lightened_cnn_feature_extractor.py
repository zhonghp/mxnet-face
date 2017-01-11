#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-11 11:01:20
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 17:44:48

import numpy as np
import mxnet as mx

import util
import config
from feature_extractor import FeatureExtractor
from lightened_cnn import lightened_cnn_b_feature


class LightenedCnnFeatureExtractor(FeatureExtractor):

    def __load_exector(self):
        _, model_args, model_auxs = mx.model.load_checkpoint(
            config.model_prefix, config.epoch)
        symbol = lightened_cnn_b_feature()
        return symbol, model_args, model_auxs

    def __init__(self):
        if config.gpu_id < 0:
            self.__ctx = mx.cpu()
        else:
            self.__ctx = mx.gpu(config.gpu_id)
        self.__symbol, self.__model_args, self.__model_auxs = self.__load_exector()

    def extract_feature(self, img_bgr):
        img_gray = util.bgr2gray(img_bgr)
        face = np.expand_dims(img_gray, axis=0) / 255.0
        assert(face.shape == (1, config.face_size, config.face_size))
        data_arr = np.zeros(
            (1, 1, config.face_size, config.face_size), dtype=float)
        data_arr[0][:] = face
        self.__model_args['data'] = mx.nd.array(data_arr, self.__ctx)
        exactor = self.__symbol.bind(self.__ctx, self.__model_args,
                                     args_grad=None, grad_req="null", aux_states=self.__model_auxs)
        exactor.forward(is_train=False)
        exactor.outputs[0].wait_to_read()
        feature = exactor.outputs[0].asnumpy()[0]
        return feature


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        img1 = '../../../data/lfw-align/Shane_Loux/Shane_Loux_0001.png'
        img2 = '../../../data/lfw-align/Val_Ackerman/Val_Ackerman_0001.png'
    else:
        img1 = sys.argv[1].strip()
        img2 = sys.argv[2].strip()
    import cv2
    config.channel_num = 1
    config.face_size = 128
    config.feature_size = 256
    config.extractor = 'lighteden_cnn'

    extractor = LightenedCnnFeatureExtractor()
    img1_bgr = cv2.imread(img1)
    img2_bgr = cv2.imread(img2)
    feat1 = extractor.extract_feature(img1_bgr)
    feat2 = extractor.extract_feature(img2_bgr)
    print util.cosine_similarity(feat1, feat2)
