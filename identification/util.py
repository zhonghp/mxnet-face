#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-11 11:06:18
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 22:29:19

import cv2
import numpy as np

import config


def bgr2gray(img_bgr):
    if img_bgr is not None:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = None
    return img_gray


def rgb2gray(img_rgb):
    if img_rgb is not None:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = None
    return img_gray


def bgr2rgb(img_bgr):
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = None
    return img_rgb


def rgb2bgr(img_rgb):
    if img_rgb is not None:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = None
    return img_bgr


def cosine_similarity(feat1, feat2):
    return np.dot(feat1, feat2) / np.linalg.norm(feat1) / np.linalg.norm(feat2)


def feature_str_to_feature(self, feature_str):
    segs = feature_str.decode('utf-8').split('\t')
    assert(len(segs) == 2)
    label = segs[0].strip()
    feature_str = segs[1].strip()
    feature_vec = [float(feature_val)
                   for feature_val in feature_str.split(',')]
    feature = np.ndarray(shape=(config.feature_size, ),
                         dtype=float, buffer=np.array(feature_vec))
    return feature, label


def feature_to_str(self, feature, label):
    feature_str = [unicode(feature_val) for feature_val in feature.flatten()]
    return u"{}\t{}".format(label, ','.join(feature_str)).encode('utf-8')
