#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 15:24:34
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-10 19:46:03

import cv2
import numpy as np
import mxnet as mx

import config
from align_face import AlignDlib
from lightened_cnn import lightened_cnn_b_feature

class FaceAPI:
  def __load_exector(self):
    _, model_args, model_auxs = mx.model.load_checkpoint(config.model_prefix, config.epoch)
    symbol = lightened_cnn_b_feature()
    return symbol, model_args, model_auxs

  def __bgr2gray(self, img_bgr):
    if img_bgr is not None:
      img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
      img_gray = None
    return img_gray

  def __rgb2gray(self, img_rgb):
    if img_rgb is not None:
      img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
      img_gray = None
    return img_gray

  def __bgr2rgb(self, img_bgr):
    if img_bgr is not None:
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
      img_rgb = None
    return img_rgb

  def __rgb2bgr(self, img_rgb):
    if img_rgb is not None:
      img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
      img_bgr = None
    return img_bgr

  def __init__(self):
    self.__align = AlignDlib(config.dlib_model_file)
    self.__symbol, self.__model_args, self.__model_auxs = self.__load_exector()

  def align_face(self, img_bgr, bbox):
    img_rgb = self.__bgr2rgb(img_bgr)
    return self.__align.align(config.face_size, img_rgb, bbox, ts=config.ts)

  def detect_all_faces(self, img_bgr):
    img_rgb = self.__bgr2rgb(img_bgr)
    return self.__align.getAllFaceBoundingBoxes(img_rgb)

  def detect_largest_face(self, img_bgr):
    img_rgb = self.__bgr2rgb(img_bgr)
    return self.__align.getLargestFaceBoundingBox(img_rgb)

  def verify(self, img1_bgr, img2_bgr):
    result = []
    bboxes1 = self.detect_all_faces(img1_bgr)
    bboxes2 = self.detect_all_faces(img2_bgr)

    pair_arr = np.zeros((2, 1, config.face_size, config.face_size), dtype=float)
    # img1 = np.expand_dims(self.__rgb2gray(img1_bgr), axis=0)
    # img2 = np.expand_dims(self.__rgb2gray(img2_bgr), axis=0)
    # pair_arr[0][:] = img1/255.0
    # pair_arr[1][:] = img2/255.0
    # self.__model_args['data'] = mx.nd.array(pair_arr, config.ctx)
    # exector = self.__symbol.bind(config.ctx, self.__model_args ,args_grad=None, grad_req="null", aux_states=self.__model_auxs)
    # exector.forward(is_train=False)
    # exector.outputs[0].wait_to_read()
    # output = exector.outputs[0].asnumpy()
    # return np.dot(output[0], output[1])/np.linalg.norm(output[0])/np.linalg.norm(output[1])

    for bbox1 in bboxes1:
      dist = []
      face1_rgb = self.align_face(img1_bgr, bbox1)
      face1_gray = self.__rgb2gray(face1_rgb)
      face1 = np.expand_dims(face1_gray, axis=0)
      assert(face1.shape == (1, config.face_size, config.face_size))
      pair_arr[0][:] = face1/255.0
      for bbox2 in bboxes2:
        face2_rgb = self.align_face(img2_bgr, bbox2)
        face2_gray = self.__rgb2gray(face2_rgb)
        face2 = np.expand_dims(face2_gray, axis=0)
        assert(face2.shape == (1, config.face_size, config.face_size))
        pair_arr[1][:] = face2/255.0

        self.__model_args['data'] = mx.nd.array(pair_arr, config.ctx)
        exector = self.__symbol.bind(config.ctx, self.__model_args ,args_grad=None, grad_req="null", aux_states=self.__model_auxs)
        exector.forward(is_train=False)
        exector.outputs[0].wait_to_read()
        output = exector.outputs[0].asnumpy()
        dis = np.dot(output[0], output[1])/np.linalg.norm(output[0])/np.linalg.norm(output[1])
        dist.append(dis)
      result.append(dist)
    return result


if __name__ == '__main__':
  import sys
  if len(sys.argv) != 3:
    print 'Usage: python face_api.py [image_1] [image_2]'
    sys.exit(-1)

  file1 = sys.argv[1].strip()
  file2 = sys.argv[2].strip()
  img1 = cv2.imread(file1)
  img2 = cv2.imread(file2)
  faceAPI = FaceAPI()

  # test verification
  print faceAPI.verify(img1, img2)

  # test detection
  img1_faces = faceAPI.detect_all_faces(img1)
  img2_faces = faceAPI.detect_all_faces(img2)
  for face in img1_faces:
    cv2.rectangle(img1, (face.left(), face.top()), (face.right(), face.bottom()), (55,255,155), 1)
  for face in img2_faces:
    cv2.rectangle(img2, (face.left(), face.top()), (face.right(), face.bottom()), (55,255,155), 1)
  cv2.imwrite('test1_detected.jpg', img1)
  cv2.imwrite('test2_detected.jpg', img2)

  # test alignment
  img1_face = faceAPI.detect_largest_face(img1)
  img2_face = faceAPI.detect_largest_face(img2)
  face1_rgb = faceAPI.align_face(img1, img1_face)
  face2_rgb = faceAPI.align_face(img2, img2_face)
  cv2.imwrite('test1_aligned.jpg', cv2.cvtColor(face1_rgb, cv2.COLOR_RGB2BGR))
  cv2.imwrite('test2_aligned.jpg', cv2.cvtColor(face2_rgb, cv2.COLOR_RGB2BGR))
