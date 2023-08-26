import cv2  
import numpy as np
from tensorflow import keras

detector=cv2.CascadeClassifier("C:\\Users\\devan\\OneDrive\\Documents\\Python\\haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model=keras.models.load_model("C:\\Users\\devan\\OneDrive\\Documents\\Python\\saved_mode_face_detection")
while True:
    ret,frame=cap.read()
        
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    detections=detector.detectMultiScale(gray_frame,scaleFactor=1.07,minSize=(150,150),minNeighbors=5,maxSize=(325,325))
    if type(detections)==np.ndarray:        
        for x,y,w,h in detections:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi = frame[y:y+h, x:x+w]
            
            preprocessed_frame=cv2.resize(roi,(256,256))
            preprocessed_frame=preprocessed_frame/255.0
            preprocessed_frame=np.expand_dims(preprocessed_frame,axis=0)
            
            predictions=model.predict(preprocessed_frame,verbose=0)
            
            result=np.argmax(predictions)
            if result==0:
                text='Incorrect Mask, wear it Correctly'
            elif result==1:
                text='Nice Mask! Masked'
            elif result==2:
                text='Wear a Mask'  
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break