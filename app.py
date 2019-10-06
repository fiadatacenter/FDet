from flask import Flask, request, Response,render_template,send_file
import time
from flask_cors import CORS
from flask import Flask, render_template, request
import cv2
import imageio
import requests
import cv2
import json
import os
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt


PATH_TO_TEST_IMAGES_DIR = './images'


app = Flask(__name__)
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
def detect(image):
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model",
    "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    detector.setInput(imageBlob)
    detections = detector.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.67:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    print(image)        
    return image
"""@app.route('/')
def index():
    return render_template("getImage.html")
    #return Response(open('./static/getImage.html').read(), mimetype="text/html")
"""
# save the image as a picture
@app.route('/image', methods=['POST'])
#@crossdomain(origin='*',headers=['access-control-allow-origin','Content-Type'])
def image():

    i = request.files['image']  # get the image
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (PATH_TO_TEST_IMAGES_DIR, f))
    image=cv2.imread(PATH_TO_TEST_IMAGES_DIR+"/"+f)##reading the image from html form
    image = imutils.resize(image, width=600)
    
    print("writing image")##checkpoint for debugging errors
    







    frame=detect(image)##The actual Detection function
    filename="1"+f
    print(filename)

    cv2.imwrite("static/"+filename,frame)
    return send_file("static/"+filename, mimetype='image/gif')
    #return render_template("results.html",image_name=f)

    #return Response("%s saved" % f)

if __name__ == '__main__':
    app.run()
