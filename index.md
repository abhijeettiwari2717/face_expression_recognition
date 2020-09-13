## Welcome to Face_Expression_Recogntion
# Facial Expression Recognition
## Using Python modules OpenCV, Tensorflow, Keras and layers of CNN
### Created by: Abhijeet Tiwari

### THERE ARE 3 FILES
- Emotion_little_vgg.h5
- Facial_Emotion_Recognition.py
- haarcascade_frontalface_default.xml

### Some of Code:
`
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
          `
To view rest of the repository and complete information/downloading the files see here: [Facial_Expression_Recognition](https://github.com/abhijeettiwari2717/face_expression_recognition)

See my Portfolio: [Abhijeet Tiwari](https://bit.ly/abhijeettiwari)
