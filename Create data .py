import cv2
import os
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
for i in range(1,4):
    for j in range (1,91):
        filename = 'D:/Python project/a - Copy/anhmoi/train/'  + str(i) + '.' +str(j) + '.jpg'
        frame = cv2.imread(filename)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fa = detector.detectMultiScale(gray, 1.1, 5)
        for(x,y,w,h) in fa:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            cv2.imwrite('dataset/anh'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])

