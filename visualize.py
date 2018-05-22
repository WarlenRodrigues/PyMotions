import cv2
import os
import argparse
import pandas as pd
import numpy as np


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Create argument parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to csv metadata file")
args = vars(ap.parse_args())

data = pd.read_csv(args['data'])
for idx, row in data.iterrows():
    filename = os.path.join(CUR_DIR, 'data', row['image'])
    face = tuple(row[1:5])
    landmarks = np.array(row[5:141]).reshape((68, 2))

    # Draw image with detected landmarks
    image = cv2.imread(filename)
    (x, y, w, h) = face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('Landmarks', image)
    if cv2.waitKey() == 27:
        break

