import numpy as np
import cv2

cam = cv2.VideoCapture('D:\Thesis\image_regis\Data\sample_video.mp4')
son = cv2.VideoCapture('D:\Thesis\image_regis\Data\pipeline_800ping1530speed_start5m.avi')
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sonar", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

while(cam.isOpened() and son.isOpened()):
    ret, frame = cam.read()
    rett, ping = son.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ping, cv2.COLOR_BGR2GRAY)

    # !
    # ?
    # # 
    # TODO 
    # *
    
    # * R 156, G 146, B 130
    # # H 37, S 16.7, V 61.2
    lower = np.array([10, 10, 10])
    upper = np.array([50, 200, 200])

    # lower = np.array([146, 136, 120])
    # upper = np.array([166, 156, 140])
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(frame, frame, mask= mask)

    frame_height, frame_width = frame.shape[:2] 
    ping_height, ping_width = ping.shape[:2]

    cv2.resizeWindow("Camera", (int(frame_width/2), int(frame_height/2)))
    cv2.resizeWindow("Mask", (int(frame_width/2), int(frame_height/2)))
    cv2.resizeWindow("Sonar", (int(ping_width/2), int(ping_height/2)))

    cv2.imshow('Camera',frame)
    cv2.imshow('Sonar', gray2)
    cv2.imshow('Mask', res)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()