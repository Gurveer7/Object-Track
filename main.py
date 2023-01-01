import cv2

cap = cv2.VideoCapture('video.mp4')

obj_detect = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)
while True:

    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    mask = obj_detect.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(25)
    if key == ord('q'):
        break
        