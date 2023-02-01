import cv2 as cv
import numpy as np
import functions
import time
import pyautogui

# Variables
count_center = count_offcenter = 0

# creating camera object
camera = cv.VideoCapture(0)

# Taking screenshot using PyAutoGui
img = pyautogui.screenshot()
# RGB colors
img = cv.cvtColor(np.array(img),
                     cv.COLOR_RGB2BGR)
# Blurring image using Gaussian Blur
img = cv.GaussianBlur(img, (15,15), cv.BORDER_DEFAULT)
#background = cv.imread("background-full.png", cv.IMREAD_COLOR)

while True:
    check, frame = camera.read()
    if check == False:
        break

    # converting frame into Gry image.
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image, face = functions.detect(frame, grey_frame)
    if face != None:

        # calling landmarks detector function.
        image, point_list = functions.detect_landmark(frame, grey_frame, face)

        right_eye_landmark_point = point_list[36:42]
        left_eye_landmark_point = point_list[42:48]

        eye_pos = (functions.EyeTracking(frame, grey_frame, right_eye_landmark_point), functions.EyeTracking(frame, grey_frame, left_eye_landmark_point))

        if eye_pos == ('Center' , 'Center'):
            count_center += 1
            count_offcenter = 0
            if count_center > 7:
                cv.destroyWindow('Display')
                count_center = 0
        if (eye_pos == ('Left', 'Left')) or (eye_pos == ('Right', 'Right')) or (eye_pos == ('Eye Closed','Eye Closes')):
            count_offcenter += 1
            count_center = 0
            if count_offcenter > 7:
                cv.imshow('Display', img)
                cv.setWindowProperty("Display", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                count_offcenter = 0
            
        # Show buffer window with camera frame with face and eyes detected
        cv.imshow('Camera', image)
    else:
        # If no face is found just show camera frame
        cv.imshow('Camera', frame)


    key = cv.waitKey(1)
    # if q is pressed on keyboard: quit
    if key == ord('q'):
        break
# closing the camera
camera.release()
# closing  all the windows
cv.destroyAllWindows()
