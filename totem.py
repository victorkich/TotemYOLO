import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np
import threading
import serial
import time
import os

cv2.namedWindow('TotemUFSM', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('TotemUFSM', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
BIAS = 0.2  # How much bigger the bias is, more sensible is the 'with mask' prediction (put a number between 0 and 1)


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = list()
    locs = list()
    preds = list()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if not face.size == 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above for loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return locs, preds


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.UMat(image)
    return cv2.warpAffine(image, M, (nW, nH))


def compute_bound():
    global locs, preds
    try:
        while not stop:
            if ret:
                # detect faces in the frame and determine if they are wearing a face mask or not
                (locs, preds) = detect_and_predict_mask(compute_frame, faceNet, maskNet)
    except:
        os.system("reboot")


def gate_control():
    global gate
    try:
        while not stop:
            data = arduino.readline()
            if data == 0:
                gate = True
            time.sleep(0.1)
    except:
        # os.system("reboot")
        pass


def draw_frame(image, c):
    ws = image.shape[0]
    hs = image.shape[1]
    image = cv2.UMat(image)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.05)), (int(hs * 0.05), int(ws * 0.2)), c, 9)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.05)), (int(hs * 0.2), int(ws * 0.05)), c, 9)
    image = cv2.line(image, (int(hs * 0.8), int(ws * 0.05)), (int(hs * 0.95), int(ws * 0.05)), c, 9)
    image = cv2.line(image, (int(hs * 0.95), int(ws * 0.05)), (int(hs * 0.95), int(ws * 0.2)), c, 9)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.95)), (int(hs * 0.2), int(ws * 0.95)), c, 9)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.8)), (int(hs * 0.05), int(ws * 0.95)), c, 9)
    image = cv2.line(image, (int(hs * 0.8), int(ws * 0.95)), (int(hs * 0.95), int(ws * 0.95)), c, 9)
    image = cv2.line(image, (int(hs * 0.95), int(ws * 0.8)), (int(hs * 0.95), int(ws * 0.95)), c, 9)
    return image


# stop running all processes
stop = False
gate = True  # False seams it is in opening process, True for already closed gate

# start arduino connection
# arduino = serial.Serial(port='USB0', baudrate=115200, timeout=0.1)

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("mask.model")

# camara capture
# cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)10/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot capture source'

ret, frame = cap.read()
frame_shape = frame.shape
frame = rotate_bound(frame, 270)
compute_frame = cv2.UMat.get(frame)

# for text in output
t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
frames = fps = count = 0
locs = list()
preds = list()
c_fps = list()
static_frame_time = time.time()
static_frame = False
post_delay_time = time.time()
post_delay = False
data = 0

cb = threading.Thread(target=compute_bound)
cb.setDaemon(True)
cb.start()

#gate_thread = threading.Thread(target=gate_control)
#gate_thread.setDaemon(True)
#gate_thread.start()

start = time.time()
while True:
    try:
        if static_frame:
            if time.time() - static_frame_time >= 3:
                static_frame = False
                post_delay_time = time.time()
                post_delay = True
        else:
            ret, frame = cap.read()
            if not ret:
                pass

            color = (0, 0, 0)
            frame = rotate_bound(frame, 270)
            compute_frame = cv2.UMat.get(frame)

            # loop over the detected face locations and their corresponding locations
            if locs:
                big_w = big_h = mask = withoutMask = startX = startY = 0

                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (s_x, s_y, e_x, e_y) = box
                    if big_h < e_y - s_y and big_w < e_x - s_x:
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred
                        big_h = e_y - s_y
                        big_w = e_x - s_x

                # clamp coordinates that are outside of the image
                startX, startY = max(startX, 0), max(startY, 0)

                # determine the class label and color we'll use to draw the bounding box and text
                label = "Com Mascara" if mask + (mask*BIAS) > withoutMask - (withoutMask*BIAS) else "Sem Mascara"
                color = (0, 255, 0) if label == "Com Mascara" else (0, 0, 255)

                if big_h >= frame_shape[1] * 0.25 and big_w >= frame_shape[0] * 0.25:
                    if not post_delay:
                        # opening the gate
                        if label == "Com Mascara" and gate:
                            # arduino.write(bytes(0, 'utf-8'))
                            static_frame = True
                            static_frame_time = time.time()
                        elif label == "Sem Mascara" and gate:
                            static_frame = True
                            static_frame_time = time.time()
                            cv2.putText(frame, "Por favor, utilize mascara!", (round(frame_shape[0]/25), round(frame_shape[1]/1.3)),
                                        cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 3)
                    else:
                        if time.time() - post_delay_time >= 3:
                            post_delay = False

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,
                            2), cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # draw the bounds of the image
            frame = cv2.UMat.get(frame)
            frame = draw_frame(frame, color)

            # CURRENT TIME SHOWING
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            # FPS PRINTING
            cv2.rectangle(frame, (0, 0), (175, 20), (0, 0, 0), -1)
            c_fps.append(1 / (time.time() - start))
            start = time.time()
            if len(c_fps) > 60:
                c_fps.pop(0)
            fps = sum(c_fps) / len(c_fps)
            cv2.putText(frame, current_time + " FPS : %3.2f" % fps, (0, t_size[1] + 2), cv2.FONT_HERSHEY_PLAIN, 1,
                        [255, 255, 255], 1)

        # show the output frame
        cv2.imshow("TotemUFSM", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            break
    except:
        # os.system("reboot")
        pass

cap.release()
cv2.destroyAllWindows()
del faceNet
del maskNet
