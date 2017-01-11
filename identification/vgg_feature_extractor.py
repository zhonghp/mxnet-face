#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 23:30:26
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 00:30:02

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

		self.__net = caffe.Net(config.vgg_model_def, config.vgg_model_weights, caffe.TEST)
		self.__transformer = caffe.io.Transformer({'data': self.__net.blobs['data'].data.shape})
		self.__transformer.set_transpose('data', (2, 0, 1))
		self.__transformer.set_mean('data', np.array(config.mean_val))
		# self.__transformer.set_raw_scale('data', 255)
		# self.__transformer.set_channel_swap('data', (2, 1, 0)) # swap channel from RGB to BGR
		self.__net.blobs['data'].reshape(config.batch_size, config.channel_num, config.face_size, config.face_size)

	def extract_feature(self, img_bgr):
		transformed_img = self.__transformer.preprocess('data', img_bgr)
		self.__net.blobs['data'].data[...] = transformed_img
		output = self.__net.forward()
		feature = self.__net.blobs['fc8'].data[0]
		return feature

if __name__ == '__main__':
	import cv2
	extractor = VggFeatureExtractor()
	img_bgr = cv2.imread('../model/vgg_face_caffe/ak.png')
	feature = extractor.extract_feature(img_bgr)
        print np.argmax(feature)
