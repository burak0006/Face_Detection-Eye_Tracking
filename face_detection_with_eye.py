import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Live Face Detection and Eye Tracking
Author: Burak Y

This repository comprises a real time face detection and eye tracking 
captured from a video feed using haar cascade classifiers with python-opencv. 
The script is well adapted to remove the outlier regions which are not actual 
faces or eyes. The cascades here are only intended to be used for education 
and training purposes. 

Haar Cascade is a machine learning object detection 
algorithm used to identify objects in an image or 
video and based on the concept of features. 
For more information https://opencv-python tutroals.readthedocs.io/en/latest/py_tutorials
/py_objdetect/py_face_detection/py_face_detection.html

You can open your camera by running the script on your terminal. It automatically starts to detect your face and eyes with a high success rate!
'''


## Importing haar cascades
## Please note that you should define your path correctly 
eye_cascade = cv2.CascadeClassifier('../../DATA/haarcascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('../../DATA/haarcascades/haarcascade_frontalface_default.xml')


#defining a function for eye detection
def detect_eye(img):
    
    eye_img = img.copy()
  
    eye_rects = eye_cascade.detectMultiScale(eye_img)
        
    return eye_rects

#defining a function for face detection
def detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img)
        
    return face_rects

cap = cv2.VideoCapture(0)

# it automatically gets width and height from a video feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
while True:
    
    # Capturing the video frame-by-frame
    ret, frame = cap.read()
  
    # Frame is converted to gray scale   
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Possible face coordinates extracted
    face_rect = detect_face(img)
    
    # Faces whose height are more likely to be face. Others discarded
    ind = np.argmax(face_rect[:,3])
    fg=list(face_rect[ind])
    
    # Face coordinates extracted
    x,y,w,h = fg
      
    # Face is constrained within a rectangular shape
    newimg = cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0,0,255),thickness=4)
    
    # The eyes are searched within upper half of the face        
    try:
        x1,y1,w1,h1 = detect_eye(frame[y:y+int(h/2),x:x+w])[0]
        x2,y2,w2,h2 = detect_eye(frame[y:y+int(h/2),x:x+w])[1]
  
    except IndexError:    
        continue
         
    # The both eyes are constrained within a rectangular shape
    first_new = cv2.rectangle(frame, (x1+x,y1+y), (x1+x+w1,y1+y+h1), (255,0,255), 4)
    second_new = cv2.rectangle(frame, (x2+x,y2+y), (x2+x+w2,y2+y+h2), (255,0,255), 4)
    
    # The line is drawed between two eyes
    cv2.line(frame,pt1=((x1+int(w1/2)+x),y1+y+int(h1/2)),pt2=(x2+x+int(w2/2),y2+y+int(h2/2)),color=(102, 255, 255),thickness=5)
    cv2.imshow('frame',second_new)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# You can release the capture
cap.release()
cv2.destroyAllWindows()
plt.show()