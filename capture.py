# Adapted from: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
import csv
import time
from PIL import Image
import glob
from Throught_Img_Folder import Throught_Img_Folder


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data')


def time_str():
	return time.strftime("%Y%m%d-%H%M%S")


print("Press ESC to close and 'c' to save an image")


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.join(CUR_DIR, 'shape_predictor_68_face_landmarks.dat')))

# Use an image database
cap = Throught_Img_Folder("Datateste")

# Use Webcam resource
#cap = cv2.VideoCapture(0)


if not os.path.isdir(DATA_DIR):
	os.mkdir(DATA_DIR)

metadata_filename = os.path.join(DATA_DIR, 'data-{}.csv'.format(time_str()))
with open(metadata_filename, 'w') as metadata:
	writer = csv.writer(metadata)
	writer.writerow(['image', 'face_x', 'face_y', 'face_width', 'face_height'] +
					['x' + str(i // 2) if i % 2 == 0 else 'y' + str((i - 1) // 2) for i in range(136)])
	running = True
	while running and cap.isOpened():
		ret, frame = cap.read()
		cv2.imshow('Camera', frame)
		key = cv2.waitKey(1)
		if key == 27:
			running = False
			cap.release()
		elif key == ord('c'):
			filename = 'data-{}.png'.format(time_str())
			full_filename = os.path.join(CUR_DIR, 'data', filename)

			# save image
			cv2.imwrite(full_filename, frame)

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# detect faces in the grayscale image
			rects = detector(gray, 1)

			largest_face = None
			largest_face_landmarks = None
			# loop over the face detections
			for (i, rect) in enumerate(rects):
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# convert dlib's rectangle to a OpenCV-style bounding box
				# [i.e., (x, y, w, h)], then draw the face bounding box
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				if largest_face is None or largest_face[2] < w:
					largest_face = (x, y, w, h)
					largest_face_landmarks = shape
			if largest_face is not None:
				writer.writerow([filename] + list(largest_face) + list(largest_face_landmarks.flatten()))

			if largest_face is not None:
				(x, y, w, h) = largest_face
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

				# loop over the (x, y)-coordinates for the facial landmarks
				# and draw them on the image
				for (x, y) in largest_face_landmarks:
					cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			# show the output image with the face detections + facial landmarks
			cv2.imshow('Detected', frame)
