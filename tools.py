import cv2
import dlib
from imutils import face_utils
import imutils
import numpy as np
from numpy.linalg import norm

def bb_to_rect(bb):
    print bb
    x= dlib.rectangle(int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[0] + bb[3]))
    print x
    return x

def detectFace(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    faces = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        print type(shape)
        faces.append((face_utils.rect_to_bb(rect), shape))

    '''detector = cv2.CascadeClassifier('/home/turja/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_roi = detector.detectMultiScale(gray, 1.3, 5)

    for face in face_roi:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        rect = bb_to_rect(face)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        faces.append((face, shape))'''

    return faces

def detectLipMovement(mouth, scale = 1):
    # take 3 points from the lips
    point_1 = np.array([mouth[6][0], mouth[6][1]])
    point_2 = np.array([mouth[3][0], mouth[3][1]])
    point_3 = np.array([mouth[9][0], mouth[9][1]])

    # create two vectors
    v1= point_2 - point_1
    v2 = point_3 - point_1

    #find angles between them
    c = np.dot(v1, v2)/norm(v1)/norm(v2)
    return np.arccos(np.clip(c, -1, 1))/scale

def detectLipMovement2(mouth, scale = 1):
    pivots = [mouth[13], mouth[14], mouth[15]]
    target = [mouth[19], mouth[18], mouth[17]]

    dist = 0
    for a, b in zip(pivots, target):
        dist = dist + norm(a - b)
    return dist/(len(mouth))/scale

def detectLipMovement3(mouth, scale = 1):
    list1 = np.concatenate((mouth[0:6], mouth[12:16]), axis=0)
    print len(list1)
    list2 = np.concatenate((mouth[7:12], mouth[16:]))
    print len(list2)
    dist = 0
    for m1 in list1:
        for m2 in list2:
            dist =  dist + norm(m1 - m2)

    return dist/len(list1)/len(list2)/scale


