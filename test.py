from yoloface import face_analysis
import cv2
import time

face = face_analysis()
cap = cv2.VideoCapture(0)
t = time.time()
while True:
    _, frame = cap.read()
    _, box, conf = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')
    if time.time() - t > 5 and not box:
        output_frame = face.show_output(frame, box, frame_status=True)
        t = time.time()
    else:
        output_frame = frame
    #output_frame = face.show_output(frame, box, frame_status=True)
    cv2.imshow('frame', output_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
