import numpy as np
import win32gui
from mss import mss
import ctypes
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import os
import time

import icsKb as kb

import sys
from PyQt5.QtWidgets import QApplication, QWidget, \
    QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QCoreApplication, QThread

# label_to_str = {0: '100000', 1: '101000', 2: '110000', 3: '101010', 4: '000000', 5: '010000', 6: '111000', 7: '001000', 8: '100100', 9: '110010', 10: '110100', 11: '101100', 12: '011000'}
label_to_str = {0: '100000', 1: '110000', 2: '101000', 3: '101010', 4: '000000', 5: '100100', 6: '010000', 7: '111000', 8: '001000', 9: '110010', 10: '011000', 11: '101100', 12: '110100'}
num_classes = len(label_to_str)
seq_len = 20

class Cnn3DModel(nn.Module):
    def __init__(self):
        super(Cnn3DModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 16)
        self.conv_layer2 = self._conv_layer_set(16, 32)
        self.conv_layer3 = self._conv_layer_set(32, 64)
        self.conv_layer4 = self._conv_layer_set(64, 128)
        self.fc1 = nn.Linear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)
    
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = (3, 3, 3), padding = (1, 0, 0)),
            nn.BatchNorm3d(num_features = out_c),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out

class Driver(QThread):
    def __init__(self, gui: QWidget):
        super().__init__(gui)
        self.gui = gui

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        self.hwnd = win32gui.FindWindow(None, "KartRider Client")
        if self.hwnd == 0:
            quit("Please run KartRider")
        self.rect = win32gui.GetWindowRect(self.hwnd)

        self.win_pos = {"top" : self.rect[1], "left" : self.rect[0], "width" : 1920, "height" : 1080}

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.is_running = False

    def load_model(self):
        save_folder = "../model/weights/"
        model_name = "model2.pt"
        save_path = os.path.join(save_folder, model_name)

        model = Cnn3DModel()
        model.load_state_dict(torch.load(save_path))

        return model

    def get_game_image(self, win_pos):
        sct = mss()
        sct_img = sct.grab(win_pos)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        with BytesIO() as f:
            img.save(f, format="JPEG")
            f.seek(0)
            img = Image.open(f)
            img.load()

        return img
    
    def run(self):
        result_string = '100000'

        preprocess = transforms.Compose([
            transforms.Resize((90, 160)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.is_running = True
        img_seq = []

        with torch.no_grad():
            while self.is_running:
                start_time = time.time()

                oldest_img = None
                cur_game_img = preprocess(self.get_game_image(self.win_pos)).to(self.device)
                
                if len(img_seq) < seq_len:
                    img_seq = img_seq + [cur_game_img]
                    continue
                elif len(img_seq) == seq_len:
                    oldest_img = img_seq[0]
                    img_seq = img_seq[1:] + [cur_game_img]
                    del oldest_img
                else:
                    print("img_seq length is ", len(img_seq))
                    exit(1)

                result = self.model(torch.stack(img_seq, dim = 1).unsqueeze(0))

                print(result.shape)
                pred = torch.argmax(result, dim = -1).item()

                result_string = label_to_str[pred]
                # result_string = f"{pred:03b}"
                
                print("추론결과 : ", result_string)
                self.gui.inputLabel.setText(f"추론결과 : {result_string}")

                kb.str2keys(result_string)

                t = time.time() - start_time
                print("실행시간1 : ", t)
                self.gui.timeLabel.setText(f"수행시간 : {t}")

                if t < 0.1:
                    time.sleep(0.1 - t)
                    print("sleeped")
                
                print("실행시간2 : ", time.time() - start_time)

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.timeLabel = QLabel("이곳에 추론시간")
        font1 = self.timeLabel.font()
        font1.setPointSize(20)
        self.timeLabel.setFont(font1)

        self.inputLabel = QLabel("이곳에 추론결과")
        font2 = self.inputLabel.font()
        font2.setPointSize(20)
        self.inputLabel.setFont(font2)

        startButton = QPushButton("Start")
        stopButton = QPushButton("Quit")

        startButton.clicked.connect(self.start)
        stopButton.clicked.connect(self.quit)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(startButton)
        hbox.addWidget(stopButton)
        hbox.addStretch(1)

        vbox = QVBoxLayout()
        vbox.addWidget(self.inputLabel)
        vbox.addWidget(self.timeLabel)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.setWindowTitle("Player")
        self.move(300, 300)
        self.resize(400, 200)
        self.show()

        self.driver = Driver(self)

    def start(self):
        self.driver.start()

    def quit(self):
        self.driver.is_running = False
        time.sleep(0.1)
        kb.str2keys("000000")
        QCoreApplication.instance().quit()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec_())
    except:
        pass