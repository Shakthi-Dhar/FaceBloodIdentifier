from flask import Flask, render_template, request
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image as im

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        if request.files:
            user_input = request.files["image"]
            user_input.save(os.path.sep.join(['images', user_input.filename]))

        # load the weights
        prototxtPath = os.path.sep.join(['models', "deploy.prototxt"])
        weightsPath = os.path.sep.join(['models', "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the blood face detector model from disk
        model = load_model('models/blood_noblood_classifier.model')

        # load the input image from disk, clone it, and grab the image spatial dimensions
        image = cv2.imread(os.path.sep.join(['images', user_input.filename]))
        orig = image.copy()
        (h, w) = image.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the face detections
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > 0.6:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224,
                # and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                (blood, noblood) = model.predict(face)[0]
                # determine the class label and color we'll use to draw the bounding box and text
                label = "Blood  detected, severly injured" if blood > noblood else "NoBlood detected"
                print(label)
                color = (0, 0, 255) if label == "Blood detected, severly injured" else (0, 255, 0)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(blood, noblood) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                image = im.fromarray(image)
                image.save(os.path.sep.join(['result', user_input.filename]))
                print(user_input.filename)
            return render_template('result.html', predfloc=user_input.filename, lab = label)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
