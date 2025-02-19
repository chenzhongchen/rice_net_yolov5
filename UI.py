import sys
import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton

class RiceEarDetectorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水稻穗识别")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("选择一张图片进行识别", self)
        self.label.setGeometry(50, 50, 700, 400)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_load = QPushButton("加载图片", self)
        self.btn_load.setGeometry(50, 500, 100, 30)
        self.btn_load.clicked.connect(self.load_image)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp1/weights/best.pt', force_reload=True)  # 替换为你的模型路径

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.process_image(file_name)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)

        # 获取检测结果
        detections = results.pred[0].numpy()
        num_boxes = len(detections)

        # 绘制框框
        for *xyxy, conf, cls in detections:
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)

        # 显示框框数量
        self.label.setText(f"检测到水稻穗数量: {num_boxes}")

        # 显示图片
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(q_img))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = RiceEarDetectorApp()
    window.show()
    sys.exit(app.exec_())