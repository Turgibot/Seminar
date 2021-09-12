import cv2
left_name = "Left"
cv2.namedWindow(left_name)

right_name = "Right"
cv2.namedWindow(right_name)

while True:
    isclosed = 0
    capL = cv2.VideoCapture('Data/Videos/left_demo.avi')
    capR = cv2.VideoCapture('Data/Videos/right_demo.avi')
    cv2.moveWindow(left_name, 2700, 200)
    cv2.moveWindow(right_name, 3483, 200)
    while True:

        retL, frameL = capL.read()
        retR, frameR = capL.read()
        if not (capL.isOpened() and capR.isOpened()):
            print("Error opening video  file")
            exit(-1)
        if retL and retR:
            frameL = cv2.resize(frameL, (672, 376))
            frameR = cv2.resize(frameR, (672, 376))
            cv2.imshow(left_name, frameL)
            cv2.imshow(right_name, frameR)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                isclosed = 1
                break
        else:
            break
    # To break the loop if it is closed manually
    if isclosed:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
