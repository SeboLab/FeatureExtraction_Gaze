'''
Gaze Extraction Script

Demo extracting gaze from webcam feed, following this source: https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

Second version should be made to extract gaze from large amount of video data, and log

Third Version for pipelining and realtime
'''


import cv2
import numpy as np

print("#   #   Starting Facial Recognition    #   #")

img = cv2.imread('face.jpeg')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_faces(image, classifier):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = image[y:y + h, x:x + w]
    return frame    

def detect_eyes(image, classifier):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame) # detect eyes
    width = np.size(image, 1) # get face frame width
    height = np.size(image, 0) # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,225,255),2)
        #cv2.imshow('my image',image)
        #cv2.waitKey(0)
        if y>height/2:
            pass
        eyecenter = x+w/2
        if eyecenter < width*0.5:
            left_eye = image[y:y + h, x:x+w]
            #cv2.imshow('my image',left_eye)
            #cv2.waitKey(0)
        else:
            right_eye = image[y:y + h, x:x+w]
            #cv2.imshow('my image',right_eye)
            #cv2.waitKey(0)
    return left_eye, right_eye

def cut_eyebrows(eye_img):
    height, width = eye_img.shape[:2]
    eyebrow_h = int(height / 4)
    eye_wo_brow = eye_img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return eye_wo_brow

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cut_eyebrows(img)
    img = cv2.erode(img, None, iteratio x:x + w]
    return frame    

def detect_eyes(image, classifier):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame) # detect eyes
    width = np.size(image, 1) # get face frame width
    height = np.size(image, 0) # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        #cv2.rectangle(image,(


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()