import numpy as np
from skimage import feature
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
from scipy.stats import kurtosis, skew, entropy
import dlib


considered_points = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 20, 22, 23, 25, 27,
        28, 29, 30, 31, 32, 34, 36, 37, 39, 40, 42, 43, 45, 46,
        48, 49, 51, 52, 53, 55, 58, 63, 67]

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

    def get_lbp_features(self, image, point, length=15):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        x, y = min(111, point[0]), min(111, point[1])

        y_upper = min(y + length // 2, lbp.shape[0])
        y_low = max(0, y - length // 2)
        x_upper = min(x + length // 2, lbp.shape[1])
        x_low = max(0, x - length // 2)

        roi_lbp = lbp[y_low:y_upper, x_low:x_upper]
        roi_lbp = roi_lbp.flatten()

        (hist, _) = np.histogram(roi_lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        hist = hist.astype("float")

        val = lbp[y, x]
        mean = np.average(hist)
        var = np.sqrt(np.var(hist))
        krt = kurtosis(hist)
        skw = skew(hist)
        eng = np.sqrt(np.sum(np.power(hist, 2)))
        ent = entropy(hist)

        features = np.array([val, mean, var, krt, skw, eng, ent])
        return features


class face_feature(object):
    def __init__(self, shape):
        self.features = []
        self.shape = shape
        self.h_baseline = shape[45][0] - shape[36][0]
        self.v_baseline = shape[39][1] - shape[57][0]

    def get_shape_distance(self):
        num_points = len(considered_points)
        for i in range(num_points - 1):
            x, y = self.shape[considered_points[i] - 1]
            for j in range(i + 1, num_points):
                x1, y1 = self.shape[considered_points[j] - 1]
                diff_x = x1 - x
                diff_y = y1 - y
                distance = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2))
                self.features.append(distance)

    def get_texture_features(self, img, numPoints=12, radius=4):
        num_points = len(considered_points)
        localbp = LocalBinaryPatterns(numPoints, radius)
        for i in range(num_points):
            texture_features = localbp.get_lbp_features(img, self.shape[considered_points[i] - 1])
            for item in texture_features:
                self.features.append(item)

    def length(self):
        return len(self.features)

    def add_distance(self):
        pass


def shape_to_np(shape, dtype="int"):
    # intialize the list of (x, y) coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them to a
    # 2-tuple of (x, y)-coordiantes
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def detect_landmarks(img):
    # input a image and output the coordination of facial landmarks
    # input image  has the shape of (112, 112, 3)
    # load model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    rects = detector(img, 1)

    # there are could be multiple faces in the image
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = shape_to_np(shape)

    return shape