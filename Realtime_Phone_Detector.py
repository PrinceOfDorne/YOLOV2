import numpy as np
from darkflow.net.build import TFNet
import cv2
import time

option = {
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolov2.weights',
    'threshold' : 0.2,
    'gpu' : 1.0
}

tfnet = TFNet(option)

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('test_img.bmp', frame)

    img = cv2.imread('test_img.bmp')
    results = tfnet.return_predict(img)
    count = 0
    time.sleep(0.5)
    for result in results:
        #tl = (result['topleft']['x'], result['topleft']['y'])
        #br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        if label == 'cell phone' :
            count = count + 1
        #confidence = result['confidence']
        #text = '{}: {:.0f}%'.format(label, confidence * 100) #Confidence is a per unit value and needs to be multiplied with 100
        #frame = cv2.rectangle(frame, tl, br, color, 5)
        #frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    #cv2.imshow('frame',frame)
    #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    print(count)
key = cv2.waitKey(20)
#if key == 27: #Press Esc to Collapse
#    break
cap.release()
cv2.destroyAllWindows()
