from __future__ import division

from models import Darknet
from utils.utils import load_classes, non_max_suppression_output

import argparse

import time
import torch
import threading
import numpy as np
from torch.autograd import Variable
from datetime import datetime

import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import cv2

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN-5, cv2.WINDOW_FULLSCREEN)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
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


stop = False


def update_screen():
    global x1, x2, y1, y2, conf, cls_pred
    while not stop:
        # Black image
        f = np.zeros((x, y, 3), np.uint8)

        f[start_new_i_height: (start_new_i_height + v_height),
        start_new_i_width: (start_new_i_width + v_width)] = t_frame

        # resizing to [416x 416]
        f = cv2.UMat(f)
        f = cv2.resize(f, (opt.frame_size, opt.frame_size))
        # [BGR -> RGB]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.UMat.get(f)
        # [[0...255] -> [0...1]]
        f = np.asarray(f) / 255
        # [[3, 416, 416] -> [416, 416, 3]]
        f = np.transpose(f, [2, 0, 1])
        # [[416, 416, 3] => [416, 416, 3, 1]]
        f = np.expand_dims(f, axis=0)
        # [np_array -> tensor]
        f = torch.Tensor(f)

        # [tensor -> variable]
        f = Variable(f.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(f)
        detections = non_max_suppression_output(detections, opt.conf_thres, opt.nms_thres)

        # For each detection in detections
        detection = detections[0]
        if detection is not None:
            for xx1, yy1, xx2, yy2, conf, cls_conf, cls_pred in detection:
                # Accommodate bounding box in original frame
                xx1 = int(xx1 * mul_constant - start_new_i_width)
                yy1 = int(yy1 * mul_constant - start_new_i_height)
                xx2 = int(xx2 * mul_constant - start_new_i_width)
                yy2 = int(yy2 * mul_constant - start_new_i_height)
            #_, _, _, _, conf, cls_conf, cls_pred = detection
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2
        else:
            x1 = x2 = y1 = y2 = conf = cls_pred = 0
        time.sleep(0.1)


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        global fps, frames
        while True:
            # frame extraction
            _, org_frame = cap.read()
            org_frame = rotate_bound(org_frame, 90)
            t_frame = cv2.UMat.get(org_frame)

            # Bounding box making and setting Bounding box title
            if int(cls_pred) == 0:
                # WITH_MASK
                cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # WITHOUT_MASK
                cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(org_frame, classes[int(cls_pred)] + ": %.2f" % conf, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 2)

            # CURRENT TIME SHOWING
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            # FPS PRINTING
            cv2.rectangle(org_frame, (0, 0), (175, 20), (0, 0, 0), -1)
            cv2.putText(org_frame, current_time + " FPS : %3.2f" % fps, (0, t_size[1] + 2),
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
            org_frame = cv2.UMat.get(org_frame)

            frames += 1
            fps = frames / (time.time() - start)            

            h, w, ch = org_frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(org_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(480, 640, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 480
        self.height = 640
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_35.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--frame_size", type=int, default=416, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # checking for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.frame_size).to(device)

    # loading weights
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)  # Load weights
    else:
        model.load_state_dict(torch.load(opt.weights_path))  # Load checkpoints

    # Set in evaluation mode
    model.eval()

    # Extracts class labels from file
    classes = load_classes(opt.class_path)

    # ckecking for GPU for Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # camara capture
    cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=720, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)
    assert cap.isOpened(), 'Cannot capture source'

    print("\nPerforming object detection:")

    # Video feed dimensions
    _, frame = cap.read()
    frame = rotate_bound(frame, 90)
    t_frame = cv2.UMat.get(frame)
    v_height, v_width = t_frame.shape[:2]

    # For a black image
    x = y = v_height if v_height > v_width else v_width

    # Putting original image into black image
    start_new_i_height = int((y - v_height) / 2)
    start_new_i_width = int((x - v_width) / 2)

    # For accommodate results in original frame
    mul_constant = x / opt.frame_size

    # for text in output
    t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    frames = fps = 0
    start = time.time()
    x1 = x2 = y1 = y2 = conf = cls_pred = 0
    th = threading.Thread(target = update_screen)
    th.setDaemon(True)
    th.start()

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_()

    #cap.release()
    #cv2.destroyAllWindows()

