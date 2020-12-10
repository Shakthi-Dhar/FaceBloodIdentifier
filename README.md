# Face Blood Identifier
A Machine learning algorithm based on Tensorflow framework to detect blood on face

### Use case
1. We can use this model to detect the severity of accident by checking if the driver is bleeding
2. We can use the model in CCTV cameras to identify if someone is bleeding
3. Can be used for other face detection system

The model is deployed as an API, [click here] (https://github.com/Shakthi-Dhar/api_face_blood).

### Tech stack used
1. Python Machine learning model with Tensorflow keras framework
2. Python Flask frame work for deployment
3. Heroku for hosting the API
4. Postman to test the POST request for the API

#### How the model works
1. The image sent to the model first undergoes preprocessing like resize and etc
2. After preprocessing the model runs an algorithm to identify the face in the image
3. The blood detection runs on the face identified in the image
4. Based on the accuracy, a bounding box with percentage is added over to the image

### Example
#### Input image
<img src="https://github.com/Shakthi-Dhar/FaceBloodIdentifier/blob/main/images/pred1.jpg" width="250" height="250" />

#### Predicted output image
<img src="https://github.com/Shakthi-Dhar/FaceBloodIdentifier/blob/main/result/pred1.jpg" width="250" height="250" />
