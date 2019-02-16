# -*- coding: utf-8 -*-
"""
Created on Wed May  3 01:30:08 2017

@author: Rajath Kumar K S
"""

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
#image = cv2.imread(args["image"])
#image = imutils.resize(image, width=500)
cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    cv2.imshow("Picture Taken", frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        cv2.imwrite('taken.jpeg',frame)
        break

frame = cv2.imread("taken.jpeg")
    
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# loop over the face parts individually
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = frame.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

		# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = frame[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

		# show the particular face part
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		cv2.waitKey(0)

	# visualize all facial landmarks with a transparent overlay
	output = face_utils.visualize_facial_landmarks(frame, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)

cv2.destroyAllWindows()
cap.release()