import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
import keras
from tensorflow.keras.models import load_model
from PIL import Image

model_name= "v4_epoch-10"
model = load_model(model_name+"/model.h5")

print("Using Tensorflow backend..")

categories = {0:'0','A':'A','B':'B','C': 'C','D':'D','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N','O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z'}

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    cv2image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)


    roi = cv2image[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("new", res)

    test=cv2.resize(res, (128, 128))
    cv2.imshow("print_test", test)

    result = model.predict(test.reshape(1, 128, 128, 1))

    prediction = {
                '0':result[0][0],
                'A': result[0][1],
                'B': result[0][2],
                'C': result[0][3],
                'D': result[0][4],
                'E': result[0][5],
                'F': result[0][6],
                'G': result[0][7],
                'H': result[0][8],
                'I': result[0][9],
                'J': result[0][10],
                'K': result[0][11],
                'L': result[0][12],
                'M': result[0][13],
                'N': result[0][14],
                'O': result[0][15],
                'P': result[0][16],
                'Q': result[0][17],
                'R': result[0][18],
                'S': result[0][19],
                'T': result[0][20],
                'U': result[0][21],
                'V': result[0][22],
                'W': result[0][23],
                'X': result[0][24],
                'Y': result[0][25],
                'Z': result[0][26]
                  }

    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    print(prediction)
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


cap.release()
cv2.destroyAllWindows()
