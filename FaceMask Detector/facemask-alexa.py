from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import boto3
from boto.s3.key import Key
def upload_to_s3(i,labfinal):

	from botocore.client import Config

	ACCESS_KEY_ID = """ Enter AWS Access Key ID"""
	ACCESS_SECRET_KEY = """ Enter AWS Secret Access Key"""
	BUCKET_NAME = """ Enter AWS Access S3 Bucket Name"""
	BUCKET_NAME1 = """ Enter AWS Access S3 Bucket Name"""

	data = open('final.png', 'rb')
	f = open('status.txt', 'w')
    
	s3 = boto3.resource(
    	's3',
    	aws_access_key_id=ACCESS_KEY_ID,
    	aws_secret_access_key=ACCESS_SECRET_KEY,
    	config=Config(signature_version='s3v4')
	)
	s3.Bucket(BUCKET_NAME).put_object(Key='final.png', Body=data)
	f.write(labfinal)
	f.close()
	f=open("status.txt",'r+')
	s3.Bucket(BUCKET_NAME1).put_object(Key='status.txt', Body=f.read())
	print ("Done")

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
p=0
labfinal=''
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "odyMask" if mask > withoutMask else "odyNo Mask"
		labfinal=label
		if labfinal=="odyMask":
			label="Mask"
		else:
			label="No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	cv2.imshow("Frame", frame)
	cv2.imwrite('final.png',frame)
	upload_to_s3(p,labfinal)
	p=p+1
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
