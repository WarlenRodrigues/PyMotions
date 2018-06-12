import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as plt
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import csv
import time

dados = pd.read_csv("data/Dataset.csv", sep=",")

dados.sort_values(by=['image'])
#---------------------Angulos--------------------------------------

ang1=[]
ang2=[]
ang3=[]
ang4=[]
ang5=[]

for index,row in dados.iterrows():
    
    #ANGULO 1
    vec1_1 = np.array([row['x63']-row['x48'], row['y63']-row['y48']])
    vec1_2 = np.array([row['x67']-row['x48'], row['y67']-row['y48']])
    angulo1 = np.math.atan2(np.linalg.det([vec1_1,vec1_2]),np.dot(vec1_1,vec1_2))
    ang1.append(angulo1)
    
    #ANGULO 2
    vec2_1 = np.array([row['x33']-row['x48'], row['y62']-row['y48']])
    vec2_2 = vec1_1
    angulo2 = np.math.atan2(np.linalg.det([vec2_1,vec2_2]),np.dot(vec2_1,vec2_2))
    ang2.append(angulo2)
    
    #ANGULO 3
    vec3_1 = np.array([row['x31']-row['x48'], row['y31']-row['y48']])
    vec3_2 = np.array([row['x54']-row['x48'], row['y54']-row['y48']])
    angulo3 = np.math.atan2(np.linalg.det([vec3_1,vec3_2]),np.dot(vec3_1,vec3_2))
    ang3.append(angulo3)
    
    #ANGULO 4
    vec4_1 = np.array([row['x54']-row['x57'], row['y54']-row['y57']])
    vec4_2 = np.array([row['x48']-row['x57'], row['y48']-row['y57']])
    angulo4 = np.math.atan2(np.linalg.det([vec4_1,vec4_2]),np.dot(vec4_1,vec4_2))
    ang4.append(angulo4)
    
    #ANGULO 5
    vec5_1 = np.array([row['x31']-row['x51'], row['y31']-row['y51']])
    vec5_2 = np.array([row['x35']-row['x51'], row['y35']-row['y51']])
    angulo5 = np.math.atan2(np.linalg.det([vec5_1,vec5_2]),np.dot(vec5_1,vec5_2))
    ang5.append(angulo5)
    
dados['angulo1'] = pd.Series(ang1)
dados['angulo2'] = pd.Series(ang2)
dados['angulo3'] = pd.Series(ang3)
dados['angulo4'] = pd.Series(ang4)
dados['angulo5'] = pd.Series(ang5)
#---------------------Treinamento-----------------------------------

X = dados.loc[:, ("angulo1", "angulo2", "angulo3", "angulo4", "angulo5")]
y = dados.loc[:, ("categoria")]

clf = OneVsRestClassifier(tree.DecisionTreeClassifier()).fit(X, y)
#---------------------Teste-----------------------------------------

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CUR_DIR = ""

DATA_DIR = os.path.join(CUR_DIR, 'data')

def time_str():
	return time.strftime("%Y%m%d-%H%M%S")

print("Press ESC to close")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.join(CUR_DIR, 'shape_predictor_68_face_landmarks.dat')))

cap = cv2.VideoCapture(1)

if not os.path.isdir(DATA_DIR):
	os.mkdir(DATA_DIR)

then = time.clock()

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
		now = time.clock()
		if key == 27:
			running = False
			cap.release()
		elif (now-then)>5:
			then = time.clock()
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

			#PREVISAO
			dados_face = list(largest_face_landmarks.flatten())

			#ANGULO 1
			vec1_1 = np.array([dados_face[126]-dados_face[96], dados_face[127]-dados_face[97]])
			vec1_2 = np.array([dados_face[134]-dados_face[96], dados_face[135]-dados_face[97]])
			angulo1 = np.math.atan2(np.linalg.det([vec1_1,vec1_2]),np.dot(vec1_1,vec1_2))
			
			#ANGULO 2
			vec2_1 = np.array([dados_face[66]-dados_face[96], dados_face[67]-dados_face[97]])
			vec2_2 = vec1_1
			angulo2 = np.math.atan2(np.linalg.det([vec2_1,vec2_2]),np.dot(vec2_1,vec2_2))

			#ANGULO 3
			vec3_1 = np.array([dados_face[62]-dados_face[96], dados_face[63]-dados_face[97]])
			vec3_2 = np.array([dados_face[108]-dados_face[96], dados_face[109]-dados_face[97]])
			angulo3 = np.math.atan2(np.linalg.det([vec3_1,vec3_2]),np.dot(vec3_1,vec3_2))
	
			#ANGULO 4
			vec4_1 = np.array([dados_face[108]-dados_face[114], dados_face[109]-dados_face[115]])
			vec4_2 = np.array([dados_face[96]-dados_face[114], dados_face[97]-dados_face[115]])
			angulo4 = np.math.atan2(np.linalg.det([vec4_1,vec4_2]),np.dot(vec4_1,vec4_2))

			#ANGULO 5
			vec5_1 = np.array([dados_face[62]-dados_face[102], dados_face[63]-dados_face[103]])
			vec5_2 = np.array([dados_face[70]-dados_face[102], dados_face[71]-dados_face[103]])
			angulo5 = np.math.atan2(np.linalg.det([vec5_1,vec5_2]),np.dot(vec5_1,vec5_2))

			d = {"angulo1":[angulo1], "angulo2":[angulo2], "angulo3":[angulo3], "angulo4":[angulo4], "angulo5":[angulo5]}

			Xt = pd.DataFrame(data=d)

			print(clf.predict(Xt))