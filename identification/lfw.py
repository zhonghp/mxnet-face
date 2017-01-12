#!/usr/bin/env python

import argparse
import logging
import os
import sys

import cv2
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

import util
import config
from vgg_feature_extractor import VggFeatureExtractor
from lightened_cnn_feature_extractor import LightenedCnnFeatureExtractor

def load_pairs(pairs_path):
    print("...Reading pairs.")
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert(len(pairs) == 6000)
    return np.array(pairs)

def pairs_info(pair, suffix):
    if len(pair) == 3:
        name1 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix)
        name2 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[2].zfill(4), suffix)
        same = 1
    elif len(pair) == 4:
        name1 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix)
        name2 = "{}/{}_{}.{}".format(pair[2], pair[2], pair[3].zfill(4), suffix)
        same = 0
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))
    return (name1, name2, same)

def read2img(root, name1, name2):
    img1_bgr = cv2.imread(os.path.join(root, name1))
    img2_bgr = cv2.imread(os.path.join(root, name2))
    return img1_bgr, img2_bgr

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def acc(predict_file):
    print("...Computing accuracy.")
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    accuracy = []
    thd = []
    with open(predict_file, "r") as f:
        predicts = f.readlines()
        predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
        for idx, (train, test) in enumerate(folds):
            logging.info("processing fold {}...".format(idx))
            best_thresh = find_best_threshold(thresholds, predicts[train])
            accuracy.append(eval_acc(best_thresh, predicts[test]))
            thd.append(best_thresh)
    return accuracy,thd

def get_predict_file(args, feature_extractor):
    assert(os.path.exists(args.lfw_align))
    pairs = load_pairs(args.pairs)
    with open(args.predict_file, 'w') as f:
        for pair in pairs:
            name1, name2, same = pairs_info(pair, args.suffix)
            logging.info("processing name1:{} <---> name2:{}".format(name1, name2))
            img1_bgr, img2_bgr = read2img(args.lfw_align, name1, name2)
            feat1 = feature_extractor.extract_feature(img1_bgr)
            feat2 = feature_extractor.extract_feature(img2_bgr)
            dis = util.cosine_similarity(feat1, feat2)
            f.write(name1 + '\t' + name2 + '\t' + str(dis) + '\t' + str(same) + '\n')

def print_result(args):
    accuracy, threshold = acc(args.predict_file)
    logging.info("10-fold accuracy is:\n{}\n".format(accuracy))
    logging.info("10-fold threshold is:\n{}\n".format(threshold))
    logging.info("mean threshold is:%.4f\n", np.mean(threshold))
    logging.info("mean is:%.4f, var is:%.4f", np.mean(accuracy), np.std(accuracy))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default="./pairs.txt",
                        help='Location of the LFW pairs file from http://vis-www.cs.umass.edu/lfw/pairs.txt')
    parser.add_argument('--lfw-align', type=str, default="./lfw-align",
                        help='The directory of lfw-align, which contains the aligned lfw images')
    parser.add_argument('--suffix', type=str, default="png",
                        help='The type of image')
    parser.add_argument('--predict-file', type=str, default='./predict.txt',
                        help='The file which contains similarity distance of every pair image given in pairs.txt')
    args = parser.parse_args()
    logging.info(args)
    if not os.path.isfile(args.pairs):
        logging.info("Error: LFW pairs (--lfwPairs) file not found.")
        logging.info("Download from http://vis-www.cs.umass.edu/lfw/pairs.txt.")
        logging.info("Default location:", "./pairs.txt")
        sys.exit(-1)
    print("Loading embeddings done")
    if not os.path.exists(args.lfw_align):
        logging.info("Error: lfw dataset not aligned.")
        logging.info("Please use ./utils/align_face.py to align lfw firstly")
        sys.exit(-1)
    if not os.path.isfile(args.predict_file):
        if config.extractor == 'vgg_face':
            feature_extractor = VggFeatureExtractor()
        else:
            feature_extractor = LightenedCnnFeatureExtractor()
        logging.info("begin generate the predict.txt.")
        get_predict_file(args, feature_extractor)
        logging.info("predict.txt has benn generated")
    print_result(args)

if __name__ == '__main__':
    main()
