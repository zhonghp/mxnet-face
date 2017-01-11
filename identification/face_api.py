#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 15:24:34
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-11 17:38:09

import os
import numpy as np

import util
import config
from align_face import AlignDlib
from vgg_feature_extractor import VggFeatureExtractor
from lightened_cnn_feature_extractor import LightenedCnnFeatureExtractor


class FaceAPI:

    def __init__(self):
        if config.extractor == 'vgg_face':
            self.__extractor = VggFeatureExtractor()
        else:
            self.__extractor = LightenedCnnFeatureExtractor()
        self.__db = []
        self.__align = AlignDlib(config.dlib_model_file)

    def __del__(self):
        self.__db = []

    def align_face(self, img_bgr, bbox):
        img_rgb = util.bgr2rgb(img_bgr)
        return self.__align.align(config.face_size, img_rgb, bbox, ts=config.ts)

    def detect_all_faces(self, img_bgr):
        img_rgb = util.bgr2rgb(img_bgr)
        return self.__align.getAllFaceBoundingBoxes(img_rgb)

    def detect_largest_face(self, img_bgr):
        img_rgb = util.bgr2rgb(img_bgr)
        return self.__align.getLargestFaceBoundingBox(img_rgb)

    def __get_face_feature(self, img_bgr, bbox):
        face_rgb = self.align_face(img_bgr, bbox)
        face_bgr = util.rgb2bgr(face_rgb)
        return self.__extractor.extract_feature(face_bgr)

    def clear_db(self):
        self.__db = []

    def __feature_str_to_feature(self, feature_str):
        segs = feature_str.decode('utf-8').split('\t')
        assert(len(segs) == 2)
        label = segs[0].strip()
        feature_str = segs[1].strip()
        feature_vec = [float(feature_val)
                       for feature_val in feature_str.split(',')]
        feature = np.ndarray(shape=(config.feature_size, ),
                             dtype=float, buffer=np.array(feature_vec))
        return feature, label

    def __add_a_feature_str_to_db(self, feature_str):
        feature_with_label = self.__feature_str_to_feature(feature_str)
        self.__db.append(feature_with_label)

    def load_db(self, filename="train.db"):
        if filename and not os.path.exists(filename):
            return
        self.clear_db()
        if not filename:
            return
        with open(filename, 'r') as reader:
            for line in reader.readlines():
                self.__add_a_feature_str_to_db(line.strip())

    def __search_db(self, feature, max_num, threshold):
        score_list = []
        for index, (db_feature, db_label) in enumerate(self.__db):
            similarity = util.cosine_similarity(feature, db_feature)
            if similarity > threshold:
                score_list.append((index, db_label, similarity))
        if len(score_list) == 0:
            return 0, []
        value_to_cmp = lambda pair: pair[2]
        sorted_score_list = sorted(
            score_list, key=value_to_cmp, reverse=True)
        if max_num > len(sorted_score_list):
            max_num = len(sorted_score_list)
        return max_num, sorted_score_list[:max_num]

    def __search_db_by_face(self, img_bgr, bbox, max_num, threshold):
        feature = self.__get_face_feature(img_bgr, bbox)
        return self.__search_db(feature, max_num, threshold)

    def search_db(self, img_bgr, max_num=3, threshold=1e-9):
        all_score_list = []
        bboxes = self.detect_all_faces(img_bgr)
        for bbox in bboxes:
            count, score_list = self.__search_db_by_face(
                img_bgr, bbox, max_num, threshold)
            all_score_list.extend(score_list)
        if len(all_score_list) == 0:
            return 0, []
        value_to_cmp = lambda pair: pair[2]
        sorted_score_list = sorted(
            all_score_list, key=value_to_cmp, reverse=True)
        if max_num > len(sorted_score_list):
            max_num = len(sorted_score_list)
        return max_num, sorted_score_list[:max_num]

    def __feature_to_str(self, feature, label):
        feature_str = [unicode(feature_val)
                       for feature_val in feature.flatten()]
        return u"{}\t{}".format(label, ','.join(feature_str)).encode('utf-8')

    def gen_feature_str_from_db(self):
        for labeled_feature in self.__db:
            yield self.__feature_to_str(*labeled_feature)

    def save_db(self, filename="train.db"):
        with open(filename, 'w') as writer:
            for feature_str in self.gen_feature_str_from_db():
                writer.write(feature_str)
                writer.write('\n')

    def add_face_to_db(self, img_bgr, bbox, label):
        if isinstance(label, str):
            label = label.decode('utf-8')
        feature = self.__get_face_feature(img_bgr, bbox)
        self.__db.append((feature, label))

    def build_db(self, root):
        self.clear_db()
        for subdir, dirs, files in os.walk(root):
            for path in files:
                (label, fname) = (os.path.basename(subdir), path)
                (name, ext) = os.path.splitext(fname)
                if ext in config.exts:
                    print label, subdir, fname
                    img_bgr = cv2.imread(os.path.join(subdir, fname))
                    bboxes = self.detect_all_faces(img_bgr)
                    for bbox in bboxes:
                        self.add_face_to_db(img_bgr, bbox, label)

    def verify(self, img1_bgr, img2_bgr):
        result = []
        bboxes1 = self.detect_all_faces(img1_bgr)
        bboxes2 = self.detect_all_faces(img2_bgr)

        for bbox1 in bboxes1:
            dist = []
            feat1 = self.__get_face_feature(img1_bgr, bbox1)
            for bbox2 in bboxes2:
                feat2 = self.__get_face_feature(img2_bgr, bbox2)
                dist.append(util.cosine_similarity(feat1, feat2))
            result.append(dist)
        return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print 'Usage: python face_api.py [image_1] [image_2]'
        sys.exit(-1)

    import cv2
    file1 = sys.argv[1].strip()
    file2 = sys.argv[2].strip()
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)

    faceAPI = FaceAPI()
    # test verification
    print faceAPI.verify(img1, img2)

    # # test detection
    # img1_faces = faceAPI.detect_all_faces(img1)
    # img2_faces = faceAPI.detect_all_faces(img2)
    # for face in img1_faces:
    #     cv2.rectangle(img1, (face.left(), face.top()),
    #                   (face.right(), face.bottom()), (55, 255, 155), 1)
    # for face in img2_faces:
    #     cv2.rectangle(img2, (face.left(), face.top()),
    #                   (face.right(), face.bottom()), (55, 255, 155), 1)
    # cv2.imwrite('test1_detected.jpg', img1)
    # cv2.imwrite('test2_detected.jpg', img2)

    # # test alignment
    # img1_face = faceAPI.detect_largest_face(img1)
    # img2_face = faceAPI.detect_largest_face(img2)
    # face1_rgb = faceAPI.align_face(img1, img1_face)
    # face2_rgb = faceAPI.align_face(img2, img2_face)
    # cv2.imwrite('test1_aligned.jpg', cv2.cvtColor(
    #     face1_rgb, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('test2_aligned.jpg', cv2.cvtColor(
    #     face2_rgb, cv2.COLOR_RGB2BGR))

    faceAPI = FaceAPI()
    faceAPI.build_db('test_db')
    faceAPI.save_db('test.db')

    faceAPI = FaceAPI()
    faceAPI.load_db('test.db')
    print faceAPI.search_db(img1)
    print faceAPI.search_db(img2)
