## Welcome to Face_Expression_Recogntion


Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown


There are three major files:

# Facial Expression Recognition
## Using Python modules OpenCV, Tensorflow, Keras and layers of CNN
### Created by: Abhijeet Tiwari

### THERE ARE 3 FILES
- Emotion_little_vgg.h5
- Facial_Emotion_Recognition.py
- haarcascade_frontalface_default.xml

`
 _from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow
import pkg_resources_
#complete path of your both the files.
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier=load_model('Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        

        #rect,face,image = face_detector(frame)
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        #make prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
© 2020`

For more information and downloading the files see here: [Facial_Expression_Recognition](https://github.com/abhijeettiwari2717/face_expression_recognition)
See my Portfolio: [Abhijeet Tiwari](https://bit.ly/abhijeettiwari)
