from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
import csv
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'GuardaCSV')

def time_str():
    return time.strftime("%Y%m%d-%H%M%S")

def list_files(folder):
    file_list = []
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            file_list.append(os.path.join(root, filename))
    return file_list

def find_files(folder, extension):
    files = list_files(folder)
    found = [f for f in files if f.endswith(extension)]
    return found

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.join(CUR_DIR, 'shape_predictor_68_face_landmarks.dat')))
jpgs = find_files(os.getcwd(), "jpg")

if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

metadata_filename = os.path.join(DATA_DIR, 'data-{}.csv'.format(time_str()))

def processa(frame,  filename_jpg):

        with open(metadata_filename, 'w') as metadata:

            writer = csv.writer(metadata)
            writer.writerow(['image', 'face_x', 'face_y', 'face_width', 'face_height'] +
                        ['x' + str(i // 2) if i % 2 == 0 else 'y' + str((i - 1) // 2) for i in range(136)])

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
                writer.writerow([filename_jpg] + list(largest_face) + list(largest_face_landmarks.flatten()))
            if largest_face is not None:
                (x, y, w, h) = largest_face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in largest_face_landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

for f in jpgs:
    print(f)
    name = os.path.basename(f)
    print(name)
    filename_csv = name.replace("jpg", "csv")
    print(filename_csv)

    frame = cv2.imread(f)
    metadata_filename = os.path.join(DATA_DIR, name+'.csv')
    processa(frame, name)
