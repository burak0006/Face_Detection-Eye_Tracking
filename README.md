# Face_Detection-Eye_Tracking

This repository comprises a real time face detection and eye tracking captured from a video feed using haar cascade classifiers with ```python-opencv```. The script is well adapted to remove the outlier regions which is not actual faces or eyes. The cascades here are only intended to be used for education and training purposes. 

Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of features. For more information, please follow the link https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

To run the code efficiently, you need to import all neccessary libraries and download haarcascades. It is designed for only one face and you are free to add more by playing around the code. The faces which show inconsistency and most likely a outlier is removed and eye detection is performed within the face boundaries later on. 

## Preparation
### Library
- PIL
- OpenCV 2
- numpy

Execute following commands for install library:
```sh
$ pip install opencv-python
$ pip install numpy
$ pip install matplotlib
```
You can run the code by simply typing ```python face_detection_with_eye.py``` It will automatically connect to your camera after granting permission. 

## Results

<img src="https://github.com/burak0006/Face_Detection-Eye_Tracking/blob/main/sample.png?raw=true" width = "512" height = "384"/>
