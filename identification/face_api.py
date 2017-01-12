#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-10 15:24:34
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-01-12 16:48:59

import db
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
        self.__db = db.DbManager()
        self.__align = AlignDlib(config.dlib_model_file)

    def align_face(self, img_bgr, bbox):
        if config.landmarks == 'outerEyesAndNose':
            landmarkIndices = AlignDlib.OUTER_EYES_AND_NOSE
        elif config.landmarks == 'innerEyesAndBottomLip':
            landmarkIndices = AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        elif config.landmarks == 'outerEyes':
            landmarkIndices = AlignDlib.OUTER_EYES
        else:
            landmarkIndices = AlignDlib.INNER_EYES_AND_NOSE_AND_LIP_CORNER
        print landmarkIndices
        img_rgb = util.bgr2rgb(img_bgr)
        return self.__align.align(config.face_size, img_rgb, bbox,
                                  ts=config.ts, landmarkIndices=landmarkIndices)

    def detect_all_faces(self, img_bgr):
        img_rgb = util.bgr2rgb(img_bgr)
        return self.__align.getAllFaceBoundingBoxes(img_rgb)

    def detect_largest_face(self, img_bgr):
        img_rgb = util.bgr2rgb(img_bgr)
        return self.__align.getLargestFaceBoundingBox(img_rgb)

    def __get_face_feature(self, img_bgr, bbox):
        face_rgb = self.align_face(img_bgr, bbox)
        face_bgr = util.rgb2bgr(face_rgb)
        return face_bgr, self.__extractor.extract_feature(face_bgr)

    def save_db(self, filename="train.pkl"):
        self.__db.save(filename)

    def load_db(self, filename="train.pkl"):
        self.__db.load(filename)

    def clear_db(self):
        self.__db.clear()

    def __search_db_by_face(self, img_bgr, bbox, max_num, threshold):
        _, feature = self.__get_face_feature(img_bgr, bbox)
        return self.__db.search_db(feature, max_num, threshold)

    def search_db(self, img_bgr, max_num=3, threshold=1e-9):
        bboxes = self.detect_all_faces(img_bgr)
        print len(bboxes), 'faces.'
        for bbox in bboxes:
            count, score_list = self.__search_db_by_face(
                img_bgr, bbox, max_num, threshold)
            yield bbox, count, score_list

    def __append_face_to_db(self, img_bgr, bbox, label):
        if isinstance(label, str):
            label = label.decode('utf-8')
        face_bgr, feature = self.__get_face_feature(img_bgr, bbox)
        self.__db.append_db(face_bgr, feature, label)

    def append_db(self, img_bgr, label):
        bboxes = self.detect_all_faces(img_bgr)
        for bbox in bboxes:
            self.__append_face_to_db(img_bgr, bbox, label)

    def build_db(self, root):
        self.clear_db()
        for subdir, dirs, files in os.walk(root):
            for path in files:
                (label, fname) = (os.path.basename(subdir), path)
                (name, ext) = os.path.splitext(fname)
                if ext in config.exts:
                    img_bgr = cv2.imread(os.path.join(subdir, fname))
                    self.append_db(img_bgr, label)

    def verify(self, img1_bgr, img2_bgr):
        result = []
        bboxes1 = self.detect_all_faces(img1_bgr)
        bboxes2 = self.detect_all_faces(img2_bgr)

        for bbox1 in bboxes1:
            dist = []
            _, feat1 = self.__get_face_feature(img1_bgr, bbox1)
            for bbox2 in bboxes2:
                _, feat2 = self.__get_face_feature(img2_bgr, bbox2)
                dist.append(util.cosine_similarity(feat1, feat2))
            result.append(dist)
        return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print 'Usage: python face_api.py [db_folder] [test_folder]'
        sys.exit(-1)

    import os
    import cv2
    import time
    # file1 = sys.argv[1].strip()
    # file2 = sys.argv[2].strip()
    # img1 = cv2.imread(file1)
    # img2 = cv2.imread(file2)

    # faceAPI = FaceAPI()
    # # test verification
    # print faceAPI.verify(img1, img2)

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

    db_folder = sys.argv[1].strip()
    test_folder = sys.argv[2].strip()
    threshold = 0.6

    # faceAPI = FaceAPI()
    # faceAPI.build_db(db_folder)
    # faceAPI.save_db('test.db')

    writer = open('scores.txt', 'w')

    faceAPI = FaceAPI()
    faceAPI.load_db('test.db')
    start = time.time()
    for subdir, dirs, files in os.walk(test_folder):
        for path in files:
            (label, fname) = (os.path.basename(subdir), path)
            (name, ext) = os.path.splitext(fname)
            if ext not in config.exts:
                continue
            print label, subdir, fname
            img_bgr = cv2.imread(os.path.join(subdir, fname))
            for idx, result in enumerate(faceAPI.search_db(img_bgr)):
                bbox, count, score_list = result
                print count
                for (item, similarity) in score_list:
                    label = item.label.encode('utf-8')
                    if similarity <= threshold:
                        continue
                    print label, similarity
                    face_rgb = faceAPI.align_face(img_bgr, bbox)
                    face_bgr = util.rgb2bgr(face_rgb)
                    filename1 = name + str(idx) + '_1' + ext
                    filename2 = name + str(idx) + '_' + label + '_2' + ext
                    writer.write(filename1 + '\t' + filename2 + '\t' + str(similarity) + '\n')
                    cv2.imwrite(os.path.join(
                        'error', filename1), face_bgr)
                    cv2.imwrite(os.path.join(
                        'error', filename2), item.face_bgr)

    end = time.time()
    print end-start, 's'
    writer.flush()
    writer.close()
