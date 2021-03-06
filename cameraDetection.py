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

font = cv2.FONT_HERSHEY_SIMPLEX
text = " "

dados = pd.read_csv("DataSet2.csv", sep=",")
#---------------------Angulos--------------------------------------

ang1=[]
ang2=[]
ang3=[]
ang4=[]
ang5=[]

def angulos(row):
	#ANGULO 1
	vec1_1 = np.array([row['x63']-row['x48'], row['y63']-row['y48']])
	vec1_2 = np.array([row['x67']-row['x48'], row['y67']-row['y48']])
	angulo1 = np.math.atan2(np.linalg.det([vec1_1,vec1_2]),np.dot(vec1_1,vec1_2))

	#ANGULO 2
	vec2_1 = np.array([row['x33']-row['x48'], row['y62']-row['y48']])
	vec2_2 = vec1_1
	angulo2 = np.math.atan2(np.linalg.det([vec2_1,vec2_2]),np.dot(vec2_1,vec2_2))

	#ANGULO 3
	vec3_1 = np.array([row['x31']-row['x48'], row['y31']-row['y48']])
	vec3_2 = np.array([row['x54']-row['x48'], row['y54']-row['y48']])
	angulo3 = np.math.atan2(np.linalg.det([vec3_1,vec3_2]),np.dot(vec3_1,vec3_2))

	#ANGULO 4
	vec4_1 = np.array([row['x54']-row['x57'], row['y54']-row['y57']])
	vec4_2 = np.array([row['x48']-row['x57'], row['y48']-row['y57']])
	angulo4 = np.math.atan2(np.linalg.det([vec4_1,vec4_2]),np.dot(vec4_1,vec4_2))

	#ANGULO 5
	vec5_1 = np.array([row['x31']-row['x51'], row['y31']-row['y51']])
	vec5_2 = np.array([row['x35']-row['x51'], row['y35']-row['y51']])
	angulo5 = np.math.atan2(np.linalg.det([vec5_1,vec5_2]),np.dot(vec5_1,vec5_2))

	return angulo1,angulo2,angulo3,angulo4,angulo5

for index,row in dados.iterrows():
	angulo1,angulo2,angulo3,angulo4,angulo5 = angulos(row)
	ang1.append(angulo1)
	ang2.append(angulo2)
	ang3.append(angulo3)
	ang4.append(angulo4)
	ang5.append(angulo5)

dados['angulo1'] = pd.Series(ang1)
dados['angulo2'] = pd.Series(ang2)
dados['angulo3'] = pd.Series(ang3)
dados['angulo4'] = pd.Series(ang4)
dados['angulo5'] = pd.Series(ang5)
#---------------------Treinamento-----------------------------------

X = dados.loc[:, ("angulo1", "angulo2", "angulo3", "angulo4", "angulo5")]
y = dados.loc[:, ("Categoria")]

clf = OneVsRestClassifier(GaussianNB()).fit(X, y)
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

running = True
while running and cap.isOpened():
	ret, frame = cap.read()
	#cv2.imshow('Camera', frame)
	key = cv2.waitKey(1)
	now = time.clock()
	if key == 27:
		running = False
		cap.release()
	elif (now-then)>0.5:
		text = " "
		then = time.clock()
		filename = 'data-{}.png'.format(time_str())
		full_filename = os.path.join(CUR_DIR, 'data', filename)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale image
		rects = detector(gray, 1)

		largest_face = None
		largest_face_landmarks = []
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

		if largest_face_landmarks!=[]:

			#PREVISAO
			dados_face = list(largest_face_landmarks.flatten())

			d = {}

			for i in range(len(dados_face)):
				if i%2==0:
					d["x"+str(i//2)] = dados_face[i]
				elif i%2!=0:
					d["y"+str((i-1)//2)] = dados_face[i]

			dados_atuais = pd.Series(data=d)

			angulo1,angulo2,angulo3,angulo4,angulo5 = angulos(dados_atuais)
									
			d = {"angulo1":[angulo1], "angulo2":[angulo2], "angulo3":[angulo3], "angulo4":[angulo4], "angulo5":[angulo5]}

			Xt = pd.DataFrame(data=d)

			if clf.predict(Xt)==0:
				text = "Neutro"
			if clf.predict(Xt)==1:
				text = "Feliz"
			if clf.predict(Xt)==2:
				text = "Triste"

	cv2.putText(frame,text,(0,450), font, 2,(255,255,255),2,cv2.LINE_AA)

	cv2.imshow('Camera', frame)
