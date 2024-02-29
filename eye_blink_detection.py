from imutils.video import VideoStream
import cv2
import time
import f_detector
import imutils
import numpy as np
import subprocess

# instantiate detector
detector = f_detector.eye_blink_detector()
print("Analyzing blinking rate of subject...")
# start variables for the blink detector
COUNTER = 0
TOTAL = 0

# ----------------------------- video -----------------------------
# ingest data
vs = VideoStream(src=0).start()
start_time = time.time()
while True:
    current_time = time.time()
    im = vs.read()
    im = cv2.flip(im, 1)
    im = imutils.resize(im, width=720)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detect_face    
    rectangles = detector.detector_faces(gray, 0)
    boxes_face = f_detector.convert_rectangles2array(rectangles,im)
    if len(boxes_face)!=0:
        # select the face with the most area
        areas = f_detector.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = np.expand_dims(boxes_face[index],axis=0)
        # blinks_detector
        prev_total = TOTAL
        COUNTER,TOTAL = detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
        if TOTAL > prev_total:
            print('blink detected')
        # add bounding box
        img_post = f_detector.bounding_box(im,boxes_face,['blinks: {}'.format(TOTAL)])
    else:
        img_post = im 
    # visualization 
    end_time = time.time() - current_time    
    FPS = 1/end_time
    cv2.putText(img_post,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    # cv2.imshow('blink_detection',img_post)  # Commented out this line
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
    # check if a minute has passed
    if current_time - start_time >= 60:
        print('Total blinks in last minute:', TOTAL)
        if TOTAL < 15:
            print('Blinking rate is unhealthy')
            subprocess.run(['notify-send', 'Blinking rate is unhealthy! Blink more'])
        if TOTAL > 30:
            #user is probably looking at phone or distracted, notify them to focus
            print('Were you looking somewhere else? Focus!')
            subprocess.run(['notify-send', 'Were you looking somewhere else? Focus!'])
        # reset the start time and total blinks
        start_time = current_time
        TOTAL = 0