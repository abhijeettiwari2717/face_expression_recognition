## Welcome to Face_Expression_Recogntion


### Markdown


There are three major files:

# Facial Expression Recognition
## Using Python modules OpenCV, Tensorflow, Keras and layers of CNN
### Created by: Abhijeet Tiwari

### THERE ARE 3 FILES
- Emotion_little_vgg.h5
- Facial_Emotion_Recognition.py
- haarcascade_frontalface_default.xml

`from keras.models import load_model`
`from time import sleep`
`from keras.preprocessing.image import img_to_array`
`from keras.preprocessing import image`
`import cv2`
`import numpy as np`
`import tensorflow`
`import pkg_resources_`
`face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')`
`classifier=load_model('Emotion_little_vgg.h5')`

`class_labels = ['Angry','Happy','Neutral','Sad','Surprise']`

To view rest of the project and complete information/downloading the files see here: [Facial_Expression_Recognition](https://github.com/abhijeettiwari2717/face_expression_recognition)
See my Portfolio: [Abhijeet Tiwari](https://bit.ly/abhijeettiwari)
