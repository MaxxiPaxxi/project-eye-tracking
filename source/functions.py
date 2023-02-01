import cv2 as cv
import numpy as np
import dlib
import math

# face detector object
face_detect = dlib.get_frontal_face_detector()

# landmarks detector
predictor = dlib.shape_predictor("external/shape_predictor_68_face_landmarks.dat")

# creating face detector function
def detect(image, grey):
    cord_face1 = (0, 0)
    cord_face2 = (0, 0)
    # getting faces from face detector
    faces = face_detect(grey)
    face = None
    # Looping over all faces
    for face in faces:
        # getting coordinates of face.
        cord_face1 = (face.left(), face.top())
        cord_face2 = (face.right(), face.bottom())

    cv.rectangle(image, cord_face1, cord_face2, (0,255,0), 2)
    return image, face


def detect_landmark(image, grey, face):
    # Calling Shape Predictor
    landmarks = predictor(grey, face)
    point_list = []
    # Looping over the almost 70 landmark points in the face
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)

        # getting x and y coordinates of each mark and adding into list.
        point_list.append(point)
        # draw if draw is True.
        # draw circle on each landmark
    cv.circle(image, point, 3, (0,0,255), 1)
    return image, point_list

# Eyes Tracking function.

def EyeTracking(image, grey, eyePoints):
    # getting dimensions of image
    dim = grey.shape
    # creating mask .
    mask = np.zeros(dim, dtype=np.uint8)
    # converting eyePoints into Numpy arrays.
    PollyPoints = np.array(eyePoints, dtype=np.int32)
    # Filling the Eyes portion with WHITE color.
    cv.fillPoly(mask, [PollyPoints], 255)

    # Writing grey image where color is White  in the mask using Bitwise and operator.
    eyeImage = cv.bitwise_and(grey, grey, mask=mask)

    # getting the max and min points of eye inorder to crop the eyes from Eye image .
    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    # other then eye area will black, making it white
    eyeImage[mask == 0] = 255

    # cropping the eye form eyeImage.
    cropedEye = eyeImage[minY:maxY, minX:maxX]

    # getting width and height of cropedEye
    height, width = cropedEye.shape

    divPart = int(width/3)

    #  applying the threshold to the eye .
    ret, thresholdEye = cv.threshold(cropedEye, 100, 255, cv.THRESH_BINARY)

    # dividing the eye into Three parts .
    rightPart = thresholdEye[0:height, 0:divPart]
    centerPart = thresholdEye[0:height, divPart:divPart+divPart]
    leftPart = thresholdEye[0:height, divPart+divPart:width]

    # counting Black pixel in each part using numpy.
    rightBlackPx = np.sum(rightPart == 0)
    centerBlackPx = np.sum(centerPart == 0)
    leftBlackPx = np.sum(leftPart == 0)
    pos = Position([rightBlackPx, centerBlackPx, leftBlackPx])

    return pos


def Position(ValuesList):

    maxIndex = ValuesList.index(max(ValuesList))
    posEye = ''
    if maxIndex == 0:
        posEye = "Right"
    elif maxIndex == 1:
        posEye = "Center"
    elif maxIndex == 2:
        posEye = "Left"
    else:
        posEye = "Eye Closed"
    return posEye
