import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import datetime
import time
import concurrent.futures
import threading
import tkinter.ttk as ttk
from datetime import datetime
from PIL import ImageTk, Image
import os
from tkinter import messagebox

import Lane_Recovery_run as LaneRecovery

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import random
import torch
import socket
import serial
import serial.tools.list_ports

import queue
import threading

import Timeserise_transformer_v6_Run
import Lane_Tracking_v5_Run

import Tracking_Extract_and_detect_color1
import Tracking_Extract_and_detect_color2

from torchvision.transforms.functional import to_pil_image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

thread_local = threading.local()


# Serial COM
class SerialThread(threading.Thread):
    # seq = serial.Serial('COM5', 115200, timeout=1)  # MS-Windows
    # seq = serial.Serial('/dev/ttyUSB0', 9600) # Linux
    is_run = True

    def __init__(self, que, port, baudrate, stopbit, parity, databit):
        print("Init!!")

        stopbitValue = serial.STOPBITS_ONE
        if stopbit == str(1):
            stopbitValue = serial.STOPBITS_ONE
        elif stopbit == str(1.5):
            stopbitValue = serial.STOPBITS_ONE_POINT_FIVE
        elif stopbit == str(2):
            stopbitValue = serial.STOPBITS_TWO

        parityValue = serial.PARITY_NONE
        if parity == "NONE":
            parityValue = serial.PARITY_NONE
        elif parity == "ODD":
            parityValue = serial.PARITY_ODD
        elif parity == "EVEN":
            parityValue = serial.PARITY_EVEN
        elif parity == "MARK":
            parityValue = serial.PARITY_MARK
        elif parity == "SPACE":
            parityValue = serial.PARITY_SPACE

        self.seq = serial.Serial(port=port, baudrate=baudrate, stopbits=stopbitValue, parity=parityValue, bytesize=int(databit), timeout=0)  # MS-Windows

        threading.Thread.__init__(self)
        self.queue = que

    def __del__(self):
        print("객체가 소멸합니다.")
        self.is_run = False

    def run(self):
        while self.is_run:
            if self.seq.inWaiting() > 0:
                text = self.seq.readline(self.seq.inWaiting())
                print(text)
            # if self.seq.inWaiting():
            #     print("Wating!")
            #     text = self.seq.readline(self.seq.inWaiting())
            #     self.queue.put(text)

        self.seq.close()
        print("End!")



class App:
    def __init__(self, mainClass, window):
        self.window = window
        self.option = None
        self.is_recording = False
        self.width = 1920
        self.height = 1080
        self.resize = 1
        self.start = False

        self.fps = 13
        self.delay = 25
        self.prevTime = 0
        self.prev_time = 0
        self.directory = "avi"

        self.photo = None
        self.photo2 = None
        self.photo3 = None

        self.center1 = None
        self.center2 = None
        self.perspectivePoints1 = None
        self.perspectivePoints2 = None
        self.matrix = None

        self.frame = 0
        self.centerS_array = []
        self.prediction_array = []
        self.predictionIndex = 0

        self.colors = ('b', 'g', 'r')
        self.laneTrackingBool = False
        self.nozzleState = False
        self.beforeNozzleState = False
        self.histBool1 = True
        self.histBool2 = True
        # self.cap1 = cv2.VideoCapture('C:/Front.mp4')
        # self.cap2 = cv2.VideoCapture('C:/Back.mp4')
        self.cap1 = cv2.VideoCapture('avi/camera1_2022-10-13_10-26-28.avi')
        self.cap2 = cv2.VideoCapture('avi/camera2_2022-10-13_10-26-28.avi')
        # self.cap1 = cv2.VideoCapture(rf'C:\Users\TECH\Desktop\yuil_1108\yuil\mv\22.10.13_f_mv_1\camera1_2022-10-13_12-49-14.avi')
        # self.cap2 = cv2.VideoCapture(rf'C:\Users\TECH\Desktop\yuil_1108\yuil\mv\22.10.13_b_mv_1\camera2_2022-10-13_12-49-14.avi')

        self.txtFile = open("log/log.txt", 'w')

        self.zoom_savedVar1 = -999
        self.zoom_savedVar2 = -999
        self.focus_savedVar1 = -999
        self.focus_savedVar2 = -999
        self.brightness_savedVar1 = -999
        self.brightness_savedVar2 = -999
        self.saturation_savedVar1 = -999
        self.saturation_savedVar2 = -999
        self.hue_savedVar1 = -999
        self.hue_savedVar2 = -999
        self.contrast_savedVar1 = -999
        self.contrast_savedVar2 = -999
        self.exposure_savedVar1 = -999
        self.exposure_savedVar2 = -999


        self.calculateDistanceVal = tkinter.DoubleVar()
        self.calculateDistanceVal.set(4)

        # # TCP/IP 통신
        # server_ip = 'localhost'  # 위에서 설정한 서버 ip
        # server_port = 9999  # 위에서 설정한 서버 포트번호
        #
        # server_ip = 'localhost'  # 위에서 설정한 서버 ip
        # server_port = 3333  # 위에서 설정한 서버 포트번호
        #
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.connect((server_ip, server_port))

        # self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server.connect((server_ip, server_port))

        # Tracking Extract and detect color
        self.pcent1, self.pcent2, self.pcent3, self.pcent4, self.pcent5 = [0, 0, 0, 0, 0]
        self.state1, self.state2, self.state3, self.state4, self.state5 = [False, False, False, False, False]
        self.y_block1, self.y_block2, self.y_block3, self.y_block4, self.y_block5 = [5, 15, 35, 60, 100]
        self.track1, self.track2, self.track3, self.track4, self.track5 = [0, 0, 0, 0, 0]
        self.track = 0
        self.shiftArray1 = []
        self.shiftArray2 = []
        self.lineStart = 0
        self.count = 0

        # 로봇 이동 이전값
        self.beforeValue = 0

        # 카메라 녹화 설정
        self.frmRecording = tkinter.Frame(self.window, width=1920, height=50, highlightthickness=0)
        self.frmRecording.grid(column=0, row=0, columnspan=3)
        self.frmRecording.pack_propagate(0)
        self.frmRecording.grid_propagate(0)

        self.cameraSetting = tkinter.LabelFrame(self.frmRecording, text="카메라", relief="solid", bd=1, padx=5)
        self.cameraSetting.grid(column=0, row=0, sticky="nsew")
        self.frmRecordingStart = tkinter.LabelFrame(self.frmRecording, text="녹화", relief="solid", bd=1, padx=5)
        self.frmRecordingStart.grid(column=1, row=0, sticky="nsew")
        self.frmRecordingSetting = tkinter.LabelFrame(self.frmRecording, text="녹화 설정", relief="solid", bd=1, padx=5)
        self.frmRecordingSetting.grid(column=2, row=0, sticky="nsew")
        self.modelSetting = tkinter.LabelFrame(self.frmRecording, text="모델 설정", relief="solid", bd=1, padx=5)
        self.modelSetting.grid(column=3, row=0, sticky="nsew")
        self.distanceSetting = tkinter.LabelFrame(self.frmRecording, text="거리 설정", relief="solid", bd=1, padx=5)
        self.distanceSetting.grid(column=4, row=0, sticky="nsew")


        # 카메라 연결 설정 -> 녹화 버튼
        self.cameraConnectIcon = tkinter.PhotoImage(file=r'./image/setting.png').subsample(8, 8)
        self.cameraConnectBtn = tkinter.Button(self.cameraSetting, text='카메라 연결', image=self.cameraConnectIcon, compound=tkinter.LEFT, justify=tkinter.LEFT, command=lambda:MainClass.cnnCamera(mainClass), state=tkinter.NORMAL)
        self.cameraConnectBtn.grid(column=0, row=0, padx=5)
        self.cameraSettingIcon = tkinter.PhotoImage(file=r'./image/setting.png').subsample(8, 8)
        self.cameraSettingBtn = tkinter.Button(self.cameraSetting, text='카메라 세팅', image=self.cameraSettingIcon, compound=tkinter.LEFT, justify=tkinter.LEFT, command=self.open_cameraSetting, state=tkinter.DISABLED)
        self.cameraSettingBtn.grid(column=1, row=0, padx=5)

        # 카메라 녹화 설정 -> 녹화 버튼
        self.recodingBtn = tkinter.Button(self.frmRecordingStart, text='녹화 시작', command=self.record, state=tkinter.DISABLED)
        self.recodingEndBtn = tkinter.Button(self.frmRecordingStart, text='녹화 종료', command=self.recordEnd, state=tkinter.DISABLED)
        self.recodingBtn.grid(column=0, row=0, padx=5)
        self.recodingEndBtn.grid(column=1, row=0, padx=5)

        # 카메라 녹화 설정 -> 녹화 설정 버튼
        self.rotateVal = tkinter.IntVar()  # radio button
        self.notRotate = tkinter.Radiobutton(self.frmRecordingSetting, text='회전 없음', justify="left", value=1,
                                             variable=self.rotateVal)
        self.rotate90R = tkinter.Radiobutton(self.frmRecordingSetting, text='시계방향 90도 회전', justify="left", value=2,
                                             variable=self.rotateVal)
        self.rotate90L = tkinter.Radiobutton(self.frmRecordingSetting, text='반시계방향 90도 회전', justify="left", value=3,
                                             variable=self.rotateVal)
        self.rotateVal.set(value=3)
        self.reverseValLR = tkinter.IntVar()
        self.notReverseLR = tkinter.Radiobutton(self.frmRecordingSetting, text='좌우 반전 없음', justify="left", value=1,
                                                variable=self.reverseValLR)
        self.reverseLR = tkinter.Radiobutton(self.frmRecordingSetting, text='좌우 반전', justify="left", value=2,
                                             variable=self.reverseValLR)
        self.reverseValUD = tkinter.IntVar()
        self.notReverseUD = tkinter.Radiobutton(self.frmRecordingSetting, text='상하 반전 없음', justify="left", value=1,
                                                variable=self.reverseValUD)
        self.reverseUD = tkinter.Radiobutton(self.frmRecordingSetting, text='상하 반전', justify="left", value=2,
                                             variable=self.reverseValUD)
        self.reverseValLR.set(value=2)
        self.reverseValUD.set(value=1)
        self.notRotate.grid(column=0, row=0, sticky="w", padx=5)
        self.rotate90R.grid(column=1, row=0, sticky="w")
        self.rotate90L.grid(column=2, row=0, sticky="w")
        self.notReverseLR.grid(column=3, row=0, sticky="w", padx=5)
        self.reverseLR.grid(column=4, row=0, sticky="w")
        self.notReverseUD.grid(column=5, row=0, sticky="w", padx=5)
        self.reverseUD.grid(column=6, row=0, sticky="w")

        # 모델 설정
        self.modelVal = tkinter.IntVar()
        self.modelWhiteLine = tkinter.Radiobutton(self.modelSetting, text='흰색 차선 모델', justify="left", value=1, variable=self.modelVal)
        self.modelYellowLine = tkinter.Radiobutton(self.modelSetting, text='노란색 차선 모델', justify="left", value=2, variable=self.modelVal)
        self.modelVal.set(value=1)

        self.modelWhiteLine.grid(column=0, row=0, sticky="w", padx=5)
        self.modelYellowLine.grid(column=1, row=0, sticky="w", padx=5)

        self.distanceVal = tkinter.IntVar()
        self.distanceVal.set(4)

        self.distanceLabel = tkinter.Label(self.distanceSetting, text="Distance: ")
        self.distanceLabel.grid(column=0, row=0, padx=10)
        self.distanceEntry = tkinter.Entry(self.distanceSetting, width=3, textvariable=self.distanceVal)
        self.distanceEntry.grid(column=1, row=0)
        self.meterLabel = tkinter.Label(self.distanceSetting, text="m")
        self.meterLabel.grid(column=2, row=0, padx=3)

        self.applyBtn = tkinter.Button(self.distanceSetting, text='적용', command=self.apply_distance,
                                           state=tkinter.NORMAL)
        self.applyBtn.grid(column=3, row=0, pady=3,  sticky="nsew")

        self.formulaVal = tkinter.StringVar()
        self.formulaVal.set("value * ([4] * 1000) / 720")

        self.formulaLabel = tkinter.Label(self.distanceSetting, textvariable=self.formulaVal)
        self.formulaLabel.grid(column=4, row=0, padx=3)


        # 기본 영상
        # self.recordedVideo = tkinter.LabelFrame(self.mainFrame, text="기본 영상", relief="solid", bd=1, pady=10, padx=10)
        # self.recordedVideo.grid(column=0, row=0, p
        # ady=10, padx=10)

        # 기본 영상
        self.basicVideo = tkinter.Frame(self.window, width=640, height=960, highlightthickness=0, bg='black')
        self.basicVideo.grid(column=0, row=1, padx=1, pady=1)
        self.basicVideo.pack_propagate(0)
        self.basicVideo.grid_propagate(0)

        # 복원 영상
        self.recoveryVideo = tkinter.Frame(self.window, width=640, height=960, highlightthickness=0, bg='black')
        self.recoveryVideo.grid(column=1, row=1, padx=1, pady=1)
        self.recoveryVideo.pack_propagate(0)
        self.recoveryVideo.grid_propagate(0)

        # 기본 영상 -> 영상 구현
        self.robotController = tkinter.Frame(self.window, width=640, height=960, highlightthickness=0, bg='#F0F0F0')
        self.robotController.grid(column=2, row=1, padx=1, pady=1)
        self.robotController.pack_propagate(0)
        self.robotController.grid_propagate(0)

        # block1
        self.basicVideoFrame = tkinter.Frame(self.basicVideo, width=320, height=960, highlightthickness=0)
        self.basicVideoFrame.grid(column=0, row=0, padx=0.5, sticky="nsew")

        self.frontVideoFrame = tkinter.Frame(self.basicVideoFrame, width=320, height=480, highlightthickness=0)
        self.frontVideoFrame.grid(column=0, row=0, sticky="nsew")
        self.frontVideo = tkinter.Canvas(self.frontVideoFrame, width=261, height=464, highlightthickness=0)
        self.frontVideo.place(relx=0.5, rely=0.5, anchor="center")
        self.backVideoFrame = tkinter.Frame(self.basicVideoFrame, width=320, height=480, highlightthickness=0)
        self.backVideoFrame.grid(column=0, row=1, sticky="nsew")
        self.backVideo = tkinter.Canvas(self.backVideoFrame, width=261, height=464, highlightthickness=0)
        self.backVideo.place(relx=0.5, rely=0.5, anchor="center")

        # self.cameraSettingBar1 = tkinter.Frame(self.basicVideo, width=320, height=240)
        # self.cameraSettingBar1.grid(column=0, row=1, padx=0, pady=0, sticky="nsew")
        # self.cameraSettingBar2 = tkinter.Frame(self.basicVideo, width=320, height=240)
        # self.cameraSettingBar2.grid(column=1, row=1, padx=0, pady=0, sticky="nsew")

        # self.histogramGraphFigure, self.histogramGraphAxs = plt.subplots(1, 2, figsize=(6.4, 2.2), constrained_layout=True)
        self.histogramGraphFigure, self.histogramGraphAxs = plt.subplots(2, figsize=(3.2, 9.6), gridspec_kw={'height_ratios': [1, 1]})
        self.histogramGraphFigure.tight_layout()
        self.histogramGraphFigure.patch.set_facecolor('#F0F0F0')

        # plt.rcParams.update({'font.size': 10})

        plt.rc('xtick', labelsize=7)
        plt.rc('ytick', labelsize=7)

        # self.histogramGraphAxs[0].xaxis.set_visible(False)
        # self.histogramGraphAxs[0].yaxis.set_visible(False)
        # self.histogramGraphAxs[1].xaxis.set_visible(False)
        # self.histogramGraphAxs[1].yaxis.set_visible(False)

        self.histogramGraphFrame = tkinter.Frame(self.basicVideo, width=320, height=960, highlightthickness=0)
        self.histogramGraphFrame.grid(column=1, row=0, padx=0.5, sticky="nsew")
        self.histogramGraph = tkinter.Canvas(self.histogramGraphFrame, width=320, height=960, highlightthickness=0)
        self.histogramGraph.place(relx=0.5, rely=0.5, anchor="center")
        # block1


        # block2
        self.recoveryVideoFrame = tkinter.Frame(self.recoveryVideo, width=320, height=960, highlightthickness=0)
        self.recoveryVideoFrame.grid(column=0, row=0, padx=0.5, sticky="nsew")

        self.frontRecoveryVideoFrame = tkinter.Frame(self.recoveryVideoFrame, width=320, height=480, highlightthickness=0)
        self.frontRecoveryVideoFrame.grid(column=0, row=0, sticky="nsew")
        self.frontRecoveryVideo = tkinter.Canvas(self.frontRecoveryVideoFrame, width=261, height=464, highlightthickness=0)
        self.frontRecoveryVideo.place(relx=0.5, rely=0.5, anchor="center")
        self.backRecoveryVideoFrame = tkinter.Frame(self.recoveryVideoFrame, width=320, height=480, highlightthickness=0)
        self.backRecoveryVideoFrame.grid(column=0, row=1, sticky="nsew")
        self.backRecoveryVideo = tkinter.Canvas(self.backRecoveryVideoFrame, width=261, height=464, highlightthickness=0)
        self.backRecoveryVideo.place(relx=0.5, rely=0.5, anchor="center")

        self.xs1 = []
        self.ys1 = []

        self.xs2 = []
        self.ys2 = []

        self.xs3 = []
        self.ys3 = []

        self.fig, self.ax = plt.subplots(2, figsize=(3.2, 9.6), gridspec_kw={'height_ratios': [1, 1]})
        self.fig.tight_layout()
        self.fig.patch.set_facecolor('#F0F0F0')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.recoveryGraphFrame = tkinter.Canvas(self.recoveryVideo, width=320, height=960, highlightthickness=0)
        self.recoveryGraphFrame.grid(column=1, row=0, padx=0.5, sticky="nsew")
        # self.recoveryGraph = FigureCanvasTkAgg(self.fig, master=self.recoveryGraphFrame)
        # self.recoveryGraph.get_tk_widget().pack(fill="both", expand=1)
        # block2


        # block3

        self.robotControllerFrame = tkinter.Frame(self.robotController, width=320, height=960, highlightthickness=0)
        self.robotControllerFrame.grid(column=0, row=0, padx=0.5, sticky="nsew")

        tempList = serial.tools.list_ports.comports()
        portList = []

        for port in tempList:
            portList.append(port.device)

        self.queue1 = queue.Queue()

        # Port1 Setting
        self.controllerSetting1 = tkinter.LabelFrame(self.robotControllerFrame, text="포트 세팅1", relief="solid", bd=1, width=320, height=160)
        self.controllerSetting1.grid(column=0, row=0)

        # port
        self.frame_com1 = tkinter.LabelFrame(self.controllerSetting1, text='COM port')
        self.frame_com1.grid(row=0, column=0, padx=10)
        self.comport_val1 = tkinter.StringVar()
        self.combo_com1 = ttk.Combobox(self.frame_com1, width=8, value=self.comport_val1, state="readonly")
        self.combo_com1['values'] = portList
        self.combo_com1.current(0)
        self.combo_com1.grid(column=0, row=0, padx=10, pady=5)

        # baud rate
        self.frame_baudrate1 = tkinter.LabelFrame(self.controllerSetting1, text='Baud Rate')
        self.frame_baudrate1.grid(row=0, column=1, padx=10)
        self.baudarte_val1 = tkinter.IntVar()
        self.combo_baudrate1 = ttk.Combobox(self.frame_baudrate1, width=8, value=self.baudarte_val1, state="readonly")
        self.combo_baudrate1['values'] = (4800, 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600)
        self.combo_baudrate1.current(5)
        self.combo_baudrate1.grid(column=0, row=0, padx=10, pady=5)

        # stop bit
        self.frame_stop1 = tkinter.LabelFrame(self.controllerSetting1, text='Stop Bit')
        self.frame_stop1.grid(row=0, column=2, padx=10)
        self.stopbit_val1 = tkinter.StringVar()
        self.combo_stopbit1 = ttk.Combobox(self.frame_stop1, width=8, textvariable=self.stopbit_val1, state="readonly")
        self.combo_stopbit1['values'] = (1, 1.5, 2)
        self.combo_stopbit1.current(0)
        self.combo_stopbit1.grid(column=0, row=0, padx=10, pady=5)


        # parity
        self.frame_parity1 = tkinter.LabelFrame(self.controllerSetting1, text='Parity')
        self.frame_parity1.grid(row=0, column=3, padx=10)
        self.parity_val1 = tkinter.StringVar()
        self.combo_parity1 = ttk.Combobox(self.frame_parity1, width=8, value=self.parity_val1, state="readonly")
        self.combo_parity1['values'] = ('Even', 'Odd', 'None', 'Mark', 'Space')
        self.combo_parity1.current(2)
        self.combo_parity1.grid(column=0, row=0, padx=10, pady=5)

        # data byte size
        self.frame_data1 = tkinter.LabelFrame(self.controllerSetting1, text='Byte size')
        self.frame_data1.grid(row=0, column=4, padx=10)
        self.databit_val1 = tkinter.StringVar()
        self.combo_databit1 = ttk.Combobox(self.frame_data1, width=8, textvariable=self.databit_val1, state="readonly")
        self.combo_databit1['values'] = (4, 5, 6, 7, 8)
        self.combo_databit1.current(4)
        self.combo_databit1.grid(column=0, row=0, padx=10, pady=5)

        self.portOpenBtn1 = tkinter.Button(self.controllerSetting1, text='포트 오픈', command=self.open_port, state=tkinter.NORMAL)
        self.portOpenBtn1.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.portCloseBtn1 = tkinter.Button(self.controllerSetting1, text='포트 닫기', command=self.close_port, state=tkinter.DISABLED)
        self.portCloseBtn1.grid(row=1, column=2, columnspan=2, padx=10, pady=10, sticky="nsew")


        self.portStatusValue = tkinter.StringVar()
        self.portStatusValue.set("STATUS: close")
        self.portStatus = tkinter.Label(self.controllerSetting1, textvariable=self.portStatusValue)
        self.portStatus.grid(row=1, column=4, padx=10, pady=10, sticky="nsew")

        # Port1 Setting

        self.controllerMessage = tkinter.LabelFrame(self.robotControllerFrame, text="메시지 전송", relief="solid", bd=1, width=320, height=160)
        self.controllerMessage.grid(column=0, row=1, pady=20, sticky="nsew")

        self.messsageNum1 = tkinter.Label(self.controllerMessage, text="1: ")
        self.messsageNum1.grid(column=0, row=0, padx=10, pady=10)
        self.messageValue1 = tkinter.Entry(self.controllerMessage, width=40)
        self.messageValue1.insert(0, 'SX+55')
        self.messageValue1.grid(column=1, row=0, padx=10, pady=10)
        self.messsageComment1 = tkinter.Label(self.controllerMessage, text="X축 +이동")
        self.messsageComment1.grid(column=2, row=0, padx=5, pady=10)
        self.messageSendBtn1 = tkinter.Button(self.controllerMessage, text='전송', width=15, command=lambda: self.send_message(self.messageValue1.get()), state=tkinter.DISABLED)
        self.messageSendBtn1.grid(column=3, row=0, padx=10, pady=10)

        self.messsageNum2 = tkinter.Label(self.controllerMessage, text="2: ")
        self.messsageNum2.grid(column=0, row=1, padx=10, pady=10)
        self.messageValue2 = tkinter.Entry(self.controllerMessage, width=40)
        self.messageValue2.insert(0, 'SX-29')
        self.messageValue2.grid(column=1, row=1, padx=10, pady=10)
        self.messsageComment2 = tkinter.Label(self.controllerMessage, text="X축 -이동")
        self.messsageComment2.grid(column=2, row=1, padx=5, pady=10)
        self.messageSendBtn2 = tkinter.Button(self.controllerMessage, text='전송', width=15, command=lambda: self.send_message(self.messageValue2.get()), state=tkinter.DISABLED)
        self.messageSendBtn2.grid(column=3, row=1, padx=10, pady=10)

        self.messsageNum3 = tkinter.Label(self.controllerMessage, text="3: ")
        self.messsageNum3.grid(column=0, row=2, padx=10, pady=10)
        self.messageValue3 = tkinter.Entry(self.controllerMessage, width=40)
        self.messageValue3.insert(0, 'SZ0')
        self.messageValue3.grid(column=1, row=2, padx=10, pady=10)
        self.messsageComment3 = tkinter.Label(self.controllerMessage, text="0점 이동")
        self.messsageComment3.grid(column=2, row=2, padx=5, pady=10)
        self.messageSendBtn3 = tkinter.Button(self.controllerMessage, text='전송', width=15, command=lambda: self.send_message(self.messageValue3.get()), state=tkinter.DISABLED)
        self.messageSendBtn3.grid(column=3, row=2, padx=10, pady=10)

        self.messsageNum4 = tkinter.Label(self.controllerMessage, text="4: ")
        self.messsageNum4.grid(column=0, row=3, padx=10, pady=10)
        self.messageValue4 = tkinter.Entry(self.controllerMessage, width=40)
        self.messageValue4.insert(0, 'SV0')
        self.messageValue4.grid(column=1, row=3, padx=10, pady=10)
        self.messsageComment4 = tkinter.Label(self.controllerMessage, text="밸브 닫기")
        self.messsageComment4.grid(column=2, row=3, padx=5, pady=10)
        self.messageSendBtn4 = tkinter.Button(self.controllerMessage, text='전송', width=15, command=lambda: self.send_message(self.messageValue4.get()), state=tkinter.DISABLED)
        self.messageSendBtn4.grid(column=3, row=3, padx=10, pady=10)

        self.messsageNum5 = tkinter.Label(self.controllerMessage, text="5: ")
        self.messsageNum5.grid(column=0, row=4, padx=10, pady=10)
        self.messageValue5 = tkinter.Entry(self.controllerMessage, width=40)
        self.messageValue5.insert(0, 'SV1')
        self.messageValue5.grid(column=1, row=4, padx=10, pady=10)
        self.messsageComment5 = tkinter.Label(self.controllerMessage, text="밸브 열기")
        self.messsageComment5.grid(column=2, row=4, padx=5, pady=10)
        self.messageSendBtn5 = tkinter.Button(self.controllerMessage, text='전송', width=15, command=lambda: self.send_message(self.messageValue5.get()), state=tkinter.DISABLED)
        self.messageSendBtn5.grid(column=3, row=4, padx=10, pady=10)

        self.messsageNum6 = tkinter.Label(self.controllerMessage, text="6: ")
        self.messsageNum6.grid(column=0, row=5, padx=10, pady=10)
        self.messageValue6 = tkinter.Entry(self.controllerMessage, width=40)
        self.messageValue6.insert(0, 'SZOK')
        self.messageValue6.grid(column=1, row=5, padx=10, pady=10)
        self.messsageComment6 = tkinter.Label(self.controllerMessage, text="SZ OK")
        self.messsageComment6.grid(column=2, row=5, padx=5, pady=10)
        self.messageSendBtn6 = tkinter.Button(self.controllerMessage, text='전송', width=15,
                                              command=lambda: self.send_message(self.messageValue6.get()),
                                              state=tkinter.DISABLED)
        self.messageSendBtn6.grid(column=3, row=5, padx=10, pady=10)

        # block3

    def animate(self, centerValue1, centerValue2, drawBlockResult1, drawBlockResult2):
        centerValue1 = -70 if centerValue1 < -70 else centerValue1
        centerValue1 = 70 if centerValue1 > 70 else centerValue1
        centerValue2 = -70 if centerValue2 < -70 else centerValue2
        centerValue2 = 70 if centerValue2 > 70 else centerValue2

        self.centerS_array.append(centerValue2)

        # if len(self.centerS_array) > 100:
        #     self.predictionIndex += 1
        #
        # if len(self.centerS_array) == 110:
        #     self.centerS_array = self.centerS_array[-100:]
        #
        # if len(self.centerS_array) == 100:
        #     self.prediction_array = []
        #     self.prediction_array.extend(Timeserise_transformer_v6_Run.transformerRun(self.centerS_array))
        #     self.predictionIndex = 0

        # Add x and y to lists
        self.xs1.append(centerValue1)
        self.ys1.append(self.frame)
        self.xs2.append(centerValue2)
        self.ys2.append(self.frame)


        if self.laneTrackingBool == True:
            self.xs3.append(drawBlockResult2)
            self.ys3.append(self.frame + 10)

        # if len(self.centerS_array) >= 100:
        #     self.xs3.append(self.prediction_array[self.predictionIndex])
        #     self.ys3.append(self.frame)

        # Limit x and y lists to 20 items
        self.xs1 = self.xs1[-20:]
        self.ys1 = self.ys1[-20:]
        self.xs2 = self.xs2[-20:]
        self.ys2 = self.ys2[-20:]

        if self.laneTrackingBool == True:
            self.xs3 = self.xs3[-30:]
            self.ys3 = self.ys3[-30:]

        plt.rcParams.update({'font.size': 10})

        plt.rc('xtick', labelsize=7)
        plt.rc('ytick', labelsize=7)

        # Draw x and y lists
        self.ax[0].clear()
        self.ax[0].plot(self.xs1, self.ys1)
        self.ax[1].clear()
        self.ax[1].plot(self.xs2, self.ys2)
        self.ax[1].plot(self.xs3, self.ys3)
        self.ax[0].set_xlim([-90, 90])
        self.ax[1].set_xlim([-90, 90])
        self.ax[0].set_xticks([-70, 0, 70])
        self.ax[1].set_xticks([-70, 0, 70])

        if self.laneTrackingBool == True:
            self.ax[1].text(0.75, 0.45, round(centerValue2 - drawBlockResult2, 2), fontsize=10, transform=plt.gcf().transFigure, verticalalignment='top')

            # 이동픽셀값
            if hasattr(self, 'thread'):
                if self.thread.seq.isOpen():
                    pixelValue = round(centerValue2 - drawBlockResult2, 2)
                    # distance = float(self.calculateDistanceVal.get()) if self.calculateDistanceVal.get() != "" else float(4)
                    distance = float(self.calculateDistanceVal.get())
                    # distance = self.calculateDistanceVal.get())
                    resultValue = pixelValue * (distance * 1000) / 720

                    self.txtFile.write(str(resultValue) + '\n');

                    msg = "SX" + ("+" + str(int(round(resultValue - self.beforeValue))) if int(round(resultValue - self.beforeValue)) >= 0 else str(int(round(resultValue - self.beforeValue))))
                    self.thread.seq.write(bytes(msg + "\\n", encoding="ascii"))

                    self.beforeValue = resultValue

            # msg = str(round(centerValue2 - drawBlockResult2, 2))
            # self.socket.sendall(msg.encode(encoding='utf-8'))
            # data = self.socket.recv(100)
            # msg = data.decode()
            # print('echo msg:', msg)


        # Format plot
        plt.xticks(ha='right')
        plt.subplots_adjust(left=0.1, wspace=0.6)

        self.fig.patch.set_facecolor('#F0F0F0')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.frame += 1

    def startFilming(self, indexs, option, fps):
        self.fps = int(fps)
        if option == "CAP_DSHOW":
            self.option = cv2.CAP_DSHOW
        else:
            self.option = None
        self.indexs = indexs
        self.multiConnect(self.indexs)
        # self.width, self.height, self.resize = self.vid.get_img_size()
        # self.create_trackbar()
        self.start = True
        self.recodingBtn["state"] = tkinter.NORMAL

        if len(self.indexs) == 2:
            self.cameraSettingBtn["state"] = tkinter.NORMAL
            self.recoveryGraph = FigureCanvasTkAgg(self.fig, master=self.recoveryGraphFrame)
            self.recoveryGraph.get_tk_widget().pack(fill="both", expand=1)
            self.update2(self.indexs)

    def stopFilming(self):
        if self.start == True:
            self.start = False
            self.recodingBtn["state"] = tkinter.DISABLED
            self.webcam1.release()
            self.frontVideo.delete(self.frontVideoIma1)
            self.frontRecoveryVideo.delete(self.basicPhotoIma)
            if len(self.indexs) == 2:
                self.webcam2.release()
                self.backVideo.delete(self.backVideoIma2)
                self.backRecoveryVideo.delete(self.basicPhotoIma2)

    def connectDevice(self, index):
        if self.indexs[0] == index:
            print("connectDevice1")
            self.vid = MyVideoCapture(index, self.width, self.height, self.resize, self.option)
            self.webcam1 = self.vid.get_webcam()
        else:
            print("connectDevice2")
            self.vid2 = MyVideoCapture(index, self.width, self.height, self.resize, self.option)
            self.webcam2 = self.vid2.get_webcam()

    def multiConnect(self, index):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.connectDevice, index)

    def close_cameraSetting(self):
        self.cameraSettingModal.destroy()
        # if messagebox.askokcancel("Quit", "창을 닫으시겠습니까?"):
        #     self.cameraSettingModal.destroy()

    def open_cameraSetting(self):
        self.cameraSettingModal = tkinter.Toplevel(self.window)
        self.cameraSettingModal.protocol("WM_DELETE_WINDOW", self.close_cameraSetting)

        self.cameraSetting = tkinter.LabelFrame(self.frmRecording, text="카메라", relief="solid", bd=1, pady=3, padx=10)

        self.cameraSettingBar1 = tkinter.LabelFrame(self.cameraSettingModal, text="Front", bd=1, relief="solid")
        self.cameraSettingBar1.grid(column=0, row=1, padx=10, pady=10, sticky="nsew")
        self.cameraSettingBar2 = tkinter.LabelFrame(self.cameraSettingModal, text="Back", bd=1, relief="solid")
        self.cameraSettingBar2.grid(column=1, row=1, padx=10, pady=10, sticky="nsew")

        self.create_trackbar()

    def process_serial(self):
        while self.queue1.qsize():
            try:
                received_data = self.queue1.get()
                print("Data received" + str(received_data))
            except queue.Empty:
                pass
        self.window.after(10, self.process_serial)

    def send_message(self, message):
        print("Send Message! - " + message)

        # self.thread.seq.write(message.encode())
        self.thread.seq.write(bytes(message + "\\n", encoding="ascii"))

    def apply_distance(self):
        print("Apply!")
        self.calculateDistanceVal.set(self.distanceEntry.get())
        self.formulaVal.set("value * ([" + self.distanceEntry.get() + "] * 1000) / 720")

    # Port Open Func
    def open_port(self):
        print("Open!")
        # print(self.combo_com1.get())
        # print(self.combo_baudrate1.get())
        # print(self.combo_stopbit1.get())
        # print(self.combo_parity1.get())
        # print(self.combo_databit1.get())

        self.thread = SerialThread(self.queue1, self.combo_com1.get(), self.combo_baudrate1.get(), self.combo_stopbit1.get(), self.combo_parity1.get(), self.combo_databit1.get())

        self.thread.start()
        self.process_serial()

        if self.thread.seq.isOpen():
            self.portStatusValue.set("STATUS: open")

            self.portOpenBtn1['state'] = tkinter.DISABLED
            self.portCloseBtn1['state'] = tkinter.NORMAL
            self.messageSendBtn1['state'] = tkinter.NORMAL
            self.messageSendBtn2['state'] = tkinter.NORMAL
            self.messageSendBtn3['state'] = tkinter.NORMAL
            self.messageSendBtn4['state'] = tkinter.NORMAL
            self.messageSendBtn5['state'] = tkinter.NORMAL
            self.messageSendBtn6['state'] = tkinter.NORMAL

    def close_port(self):
        print("Close!")

        if hasattr(self, 'thread'):
            if self.thread.seq.isOpen():
                self.thread.is_run = False
                self.portStatusValue.set("STATUS: close")

                self.portOpenBtn1['state'] = tkinter.NORMAL
                self.portCloseBtn1['state'] = tkinter.DISABLED
                self.messageSendBtn1['state'] = tkinter.DISABLED
                self.messageSendBtn2['state'] = tkinter.DISABLED
                self.messageSendBtn3['state'] = tkinter.DISABLED
                self.messageSendBtn4['state'] = tkinter.DISABLED
                self.messageSendBtn5['state'] = tkinter.DISABLED
                self.messageSendBtn6['state'] = tkinter.DISABLED

    def create_trackbar(self):
        # trackbar1
        self.zoom_var1 = tkinter.IntVar()
        self.zoom_label = tkinter.Label(self.cameraSettingBar1, text="확대")
        self.zoom_label.grid(column=0, row=1)
        self.zoom_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.zoom_var1, command=self.zoom_bar1,
                                        orient="vertical", showvalue=True,
                                        tickinterval=0, from_=100, to=0, length=300, width=10)
        self.zoom_scale.grid(column=0, row=0)
        if self.zoom_savedVar1 == -999:
            self.zoom_var1 = self.webcam1.get(cv2.CAP_PROP_ZOOM)
        else:
            self.zoom_var1 = self.zoom_savedVar1
        self.zoom_scale.set(self.zoom_var1)
        self.webcam1.set(cv2.CAP_PROP_ZOOM, self.zoom_var1)


        self.focus_var1 = tkinter.IntVar()
        self.focus_label = tkinter.Label(self.cameraSettingBar1, text="초점")
        self.focus_label.grid(column=1, row=1)
        self.focus_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.focus_var1, command=self.focus_bar1,
                                         orient="vertical", showvalue=True,
                                         tickinterval=0, from_=1023, to=0, length=300, width=10)
        self.focus_scale.grid(column=1, row=0)
        if self.focus_savedVar1 == -999:
            self.focus_var1 = self.webcam1.get(cv2.CAP_PROP_FOCUS)
        else:
            self.focus_var1 = self.focus_savedVar1
        self.focus_scale.set(self.focus_var1)
        self.webcam1.set(cv2.CAP_PROP_FOCUS, self.focus_var1)


        self.brightness_var1 = tkinter.IntVar()
        self.brightness_label = tkinter.Label(self.cameraSettingBar1, text="밝기")
        self.brightness_label.grid(column=2, row=1)
        self.brightness_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.brightness_var1,
                                              command=self.brightness_bar1, orient="vertical",
                                              showvalue=True,
                                              tickinterval=0, from_=128, to=0, length=300, width=10)
        self.brightness_scale.grid(column=2, row=0)
        if self.brightness_savedVar1 == -999:
            self.brightness_var1 = self.webcam1.get(cv2.CAP_PROP_BRIGHTNESS) + 64
        else:
            self.brightness_var1 = self.brightness_savedVar1 + 64
        self.brightness_scale.set(self.brightness_var1)
        self.webcam1.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_var1)

        self.saturation_var1 = tkinter.IntVar()
        self.saturation_label = tkinter.Label(self.cameraSettingBar1, text="채도")
        self.saturation_label.grid(column=3, row=1)
        self.saturation_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.saturation_var1,
                                              command=self.saturation_bar1, orient="vertical",
                                              showvalue=True,
                                              tickinterval=0, from_=128, to=0, length=300, width=10)
        self.saturation_scale.grid(column=3, row=0)
        if self.saturation_savedVar1 == -999:
            self.saturation_var1 = self.webcam1.get(cv2.CAP_PROP_SATURATION)
        else:
            self.saturation_var1 = self.saturation_savedVar1
        self.saturation_scale.set(self.saturation_var1)
        self.webcam1.set(cv2.CAP_PROP_SATURATION, self.saturation_var1)


        self.hue_var1 = tkinter.IntVar()
        self.hue_label = tkinter.Label(self.cameraSettingBar1, text="색상")
        self.hue_label.grid(column=4, row=1)
        self.hue_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.hue_var1,
                                       command=self.hue_bar1, orient="vertical",
                                       showvalue=True,
                                       tickinterval=0, from_=4000, to=0, length=300, width=10)
        self.hue_scale.grid(column=4, row=0)
        if self.hue_savedVar1 == -999:
            self.hue_var1 = self.webcam1.get(cv2.CAP_PROP_HUE) + 2000
        else:
            self.hue_var1 = self.hue_savedVar1 + 2000
        self.hue_scale.set(self.hue_var1)
        self.webcam1.set(cv2.CAP_PROP_HUE, self.hue_var1)


        self.contrast_var1 = tkinter.IntVar()
        self.contrast_label = tkinter.Label(self.cameraSettingBar1, text="대비")
        self.contrast_label.grid(column=5, row=1)
        self.contrast_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.contrast_var1,
                                            command=self.contrast_bar1, orient="vertical",
                                            showvalue=True,
                                            tickinterval=0, from_=95, to=0, length=300, width=10)
        self.contrast_scale.grid(column=5, row=0)
        if self.contrast_savedVar1 == -999:
            self.contrast_var1 = self.webcam1.get(cv2.CAP_PROP_CONTRAST)
        else:
            self.contrast_var1 = self.contrast_savedVar1
        self.contrast_scale.set(self.contrast_var1)
        self.webcam1.set(cv2.CAP_PROP_CONTRAST, self.contrast_var1)

        self.exposure_var1 = tkinter.IntVar()
        self.exposure_label = tkinter.Label(self.cameraSettingBar1, text="노출")
        self.exposure_label.grid(column=6, row=1)
        self.exposure_scale = tkinter.Scale(self.cameraSettingBar1, variable=self.exposure_var1,
                                            command=self.exposure_bar1, orient="vertical",
                                            showvalue=True,
                                            tickinterval=0, from_=13, to=0, length=300, width=10)
        self.exposure_scale.grid(column=6, row=0)
        if self.exposure_savedVar1 == -999:
            self.exposure_var1 = self.webcam1.get(cv2.CAP_PROP_EXPOSURE) * (-1)
        else:
            self.exposure_var1 = self.exposure_savedVar1 * (-1)
        self.exposure_scale.set(self.exposure_var1)
        self.webcam1.set(cv2.CAP_PROP_EXPOSURE, self.exposure_var1)

        if len(self.indexs) == 2:
            # trackbar2
            self.zoom_var2 = tkinter.IntVar()
            self.zoom_label2 = tkinter.Label(self.cameraSettingBar2, text="확대")
            self.zoom_label2.grid(column=0, row=1)
            self.zoom_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.zoom_var2, command=self.zoom_bar2,
                                             orient="vertical", showvalue=True,
                                             tickinterval=0, from_=100, to=0, length=300, width=10)
            self.zoom_scale2.grid(column=0, row=0)
            if self.zoom_savedVar2 == -999:
                self.zoom_var2 = self.webcam2.get(cv2.CAP_PROP_ZOOM)
            else:
                self.zoom_var2 = self.zoom_savedVar2
            self.zoom_scale2.set(self.zoom_var2)
            self.webcam2.set(cv2.CAP_PROP_ZOOM, self.zoom_var2)


            self.focus_var2 = tkinter.IntVar()
            self.focus_label2 = tkinter.Label(self.cameraSettingBar2, text="초점")
            self.focus_label2.grid(column=1, row=1)
            self.focus_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.focus_var2, command=self.focus_bar2,
                                              orient="vertical", showvalue=True,
                                              tickinterval=0, from_=1023, to=0, length=300, width=10)
            self.focus_scale2.grid(column=1, row=0)
            if self.focus_savedVar2 == -999:
                self.focus_var2 = self.webcam2.get(cv2.CAP_PROP_FOCUS)
            else:
                self.focus_var2 = self.focus_savedVar2
            self.focus_scale2.set(self.focus_var2)
            self.webcam2.set(cv2.CAP_PROP_FOCUS, self.focus_var2)


            self.brightness_var2 = tkinter.IntVar()
            self.brightness_label2 = tkinter.Label(self.cameraSettingBar2, text="밝기")
            self.brightness_label2.grid(column=2, row=1)
            self.brightness_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.brightness_var2,
                                                   command=self.brightness_bar2, orient="vertical",
                                                   showvalue=True,
                                                   tickinterval=0, from_=128, to=0, length=300, width=10)
            self.brightness_scale2.grid(column=2, row=0)
            if self.brightness_savedVar2 == -999:
                self.brightness_var2 = self.webcam2.get(cv2.CAP_PROP_BRIGHTNESS) + 64
            else:
                self.brightness_var2 = self.brightness_savedVar2 + 64
            self.brightness_scale2.set(self.brightness_var2)
            self.webcam2.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_var2)


            self.saturation_var2 = tkinter.IntVar()
            self.saturation_label2 = tkinter.Label(self.cameraSettingBar2, text="채도")
            self.saturation_label2.grid(column=3, row=1)
            self.saturation_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.saturation_var2,
                                                   command=self.saturation_bar2, orient="vertical",
                                                   showvalue=True,
                                                   tickinterval=0, from_=128, to=0, length=300, width=10)
            self.saturation_scale2.grid(column=3, row=0)
            if self.saturation_savedVar2 == -999:
                self.saturation_var2 = self.webcam2.get(cv2.CAP_PROP_SATURATION)
            else:
                self.saturation_var2 = self.saturation_savedVar2
            self.saturation_scale2.set(self.saturation_var2)
            self.webcam2.set(cv2.CAP_PROP_SATURATION, self.saturation_var2)


            self.hue_var2 = tkinter.IntVar()
            self.hue_label2 = tkinter.Label(self.cameraSettingBar2, text="색상")
            self.hue_label2.grid(column=4, row=1)
            self.hue_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.hue_var2,
                                            command=self.hue_bar2, orient="vertical",
                                            showvalue=True,
                                            tickinterval=0, from_=4000, to=0, length=300, width=10)
            self.hue_scale2.grid(column=4, row=0)
            if self.hue_savedVar2 == -999:
                self.hue_var2 = self.webcam2.get(cv2.CAP_PROP_HUE) + 2000
            else:
                self.hue_var2 = self.hue_savedVar2 + 2000
            self.hue_scale2.set(self.hue_var2)
            self.webcam2.set(cv2.CAP_PROP_HUE, self.hue_var2)

            self.contrast_var2 = tkinter.IntVar()
            self.contrast_label2 = tkinter.Label(self.cameraSettingBar2, text="대비")
            self.contrast_label2.grid(column=5, row=1)
            self.contrast_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.contrast_var2,
                                                 command=self.contrast_bar2, orient="vertical",
                                                 showvalue=True,
                                                 tickinterval=0, from_=95, to=0, length=300, width=10)
            self.contrast_scale2.grid(column=5, row=0)
            if self.contrast_savedVar2 == -999:
                self.contrast_var2 = self.webcam2.get(cv2.CAP_PROP_CONTRAST)
            else:
                self.contrast_var2 = self.contrast_savedVar2
            self.contrast_scale2.set(self.contrast_var2)
            self.webcam2.set(cv2.CAP_PROP_CONTRAST, self.contrast_var2)

            self.exposure_var2 = tkinter.IntVar()
            self.exposure_label2 = tkinter.Label(self.cameraSettingBar2, text="노출")
            self.exposure_label2.grid(column=6, row=1)
            self.exposure_scale2 = tkinter.Scale(self.cameraSettingBar2, variable=self.exposure_var2,
                                                 command=self.exposure_bar2, orient="vertical",
                                                 showvalue=True,
                                                 tickinterval=0, from_=13, to=0, length=300, width=10)
            self.exposure_scale2.grid(column=6, row=0)
            if self.exposure_savedVar2 == -999:
                self.exposure_var2 = self.webcam2.get(cv2.CAP_PROP_EXPOSURE) * (-1)
            else:
                self.exposure_var2 = self.exposure_savedVar2 * (-1)
            self.exposure_scale2.set(self.exposure_var2)
            self.webcam2.set(cv2.CAP_PROP_EXPOSURE, self.exposure_var2)

    def zoom_bar1(self, value):
        value = int(value)
        self.zoom_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_ZOOM, value)
        self.histBool1 = True

    def zoom_bar2(self, value):
        value = int(value)
        self.zoom_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_ZOOM, value)
        self.histBool2 = True

    def focus_bar1(self, value):
        value = int(value)
        self.focus_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_FOCUS, value)
        self.histBool1 = True

    def focus_bar2(self, value):
        value = int(value)
        self.focus_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_FOCUS, value)
        self.histBool2 = True

    def brightness_bar1(self, value):
        value = int(value) - 64
        self.brightness_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_BRIGHTNESS, value)
        self.histBool1 = True

    def brightness_bar2(self, value):
        value = int(value) - 64
        self.brightness_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_BRIGHTNESS, value)
        self.histBool2 = True

    def saturation_bar1(self, value):
        value = int(value)
        self.saturation_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_SATURATION, value)
        self.histBool1 = True

    def saturation_bar2(self, value):
        value = int(value)
        self.saturation_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_SATURATION, value)
        self.histBool2 = True

    def hue_bar1(self, value):
        value = int(value) - 2000
        self.hue_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_HUE, value)
        self.histBool1 = True

    def hue_bar2(self, value):
        value = int(value) - 2000
        self.hue_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_HUE, value)
        self.histBool2 = True

    def contrast_bar1(self, value):
        value = int(value)
        self.contrast_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_CONTRAST, value)
        self.histBool1 = True

    def contrast_bar2(self, value):
        value = int(value)
        self.contrast_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_CONTRAST, value)
        self.histBool2 = True

    def exposure_bar1(self, value):  # 노출 -13 ~ 0
        value = int(value) * (-1)
        self.exposure_savedVar1 = value
        self.webcam1.set(cv2.CAP_PROP_EXPOSURE, value)
        self.histBool1 = True

    def exposure_bar2(self, value):  # 노출 -13 ~ 0
        value = int(value) * (-1)
        self.exposure_savedVar2 = value
        self.webcam2.set(cv2.CAP_PROP_EXPOSURE, value)
        self.histBool2 = True

    def record(self):
        self.is_recording = True
        self.recodingBtn["state"] = tkinter.DISABLED
        self.recodingEndBtn["state"] = tkinter.NORMAL

        self.current_time = datetime.now()
        format = '%Y-%m-%d_%H-%M-%S'
        current_time_str = self.current_time.strftime(format)
        fourcc = self.vid.get_fourcc()
        fps = self.fps - 0.1
        if self.rotateVal.get() == 1:
            size = (self.width, self.height)
        else:
            size = (self.height, self.width)

        try:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
        except OSError:
            print("Error: Failed to create the directory.")

        self.writer = cv2.VideoWriter(f'avi/camera1_{current_time_str}.avi', fourcc, fps, size)
        if self.writer.isOpened() == False: raise Exception("카메라1 동영상 파일 개방 안됨")

        if len(self.indexs) == 2:
            self.writer2 = cv2.VideoWriter(f'avi/camera2_{current_time_str}.avi', fourcc, fps, size)
            if self.writer2.isOpened() == False: raise Exception("카메라2 동영상 파일 개방 안됨")

    def recordEnd(self):
        self.is_recording = False
        self.recodingBtn["state"] = tkinter.NORMAL
        self.recodingEndBtn["state"] = tkinter.DISABLED
        self.writer.release()
        if len(self.indexs) == 2:
            self.writer2.release()

    def draw_text(self, img, img2, text, x, y, text_color, font_size):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        # text_color = (255, 0, 0)
        text_color_bg = (0, 0, 0)

        text_size, _ = cv2.getTextSize(text, font, font_size, font_thickness)
        text_w, text_h = text_size
        offset = 5
        cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
        cv2.rectangle(img2, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_size, text_color, font_thickness)
        cv2.putText(img2, text, (x, y + text_h + font_scale - 1), font, font_size, text_color, font_thickness)

    def draw_text_one(self, img, text, x, y, text_color, font_size):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        # text_color = (255, 0, 0)
        text_color_bg = (0, 0, 0)

        text_size, _ = cv2.getTextSize(text, font, font_size, font_thickness)
        text_w, text_h = text_size
        offset = 5
        cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_size, text_color, font_thickness)

    # def getFrame(self):
    #     self.ret, self.frame = self.vid.get_frame()
    #     self.ret2, self.frame2 = self.vid2.get_frame()
    #     self.window.after(20, self.getFrame)
    #
    # def multiStart(self):
    #     self.window.after(0, self.getFrame)
    #     self.window.after(0, self.update)
    #     # self.getFrame() # 두 함수가 동시에 돌면 화면 렌더링이 느려짐
    #     # self.update()
    #
    # def update(self):
    #     # print('update')
    #     current_time = time.time() - self.prev_time # 경과시간 = 현재시간 - 이전 프레임 재생시간
    #
    #     # if (self.ret is True) and (current_time > 1./self.fps): # 경과시간 > 1./FPS
    #     if self.ret:
    #         if current_time > 1./self.fps: # fps 최대 제한
    #             pass
    #         else:
    #             while(True):
    #                 current_time = time.time() - self.prev_time
    #                 if current_time > 1./self.fps:
    #                     break
    #
    #         self.prev_time = time.time()
    #
    #         # fps 시간계산
    #         curTime = time.time()
    #         sec = curTime - self.prevTime
    #         self.prevTime = curTime
    #         fps = 1 / (sec)
    #         x = 30
    #         y = 80
    #         str = "FPS : %0.2f" % fps
    #
    #         self.draw_text(self.frame, self.frame2, str, x, y, (0, 255, 0), 0.7)
    #
    #         if self.is_recording:
    #             self.vid.video.write(self.frame)
    #             self.vid2.video.write(self.frame2)
    #
    #         image = PIL.Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
    #         self.photo = PIL.ImageTk.PhotoImage(image=image)
    #         self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
    #
    #         image2 = PIL.Image.fromarray(cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB))
    #         self.photo2 = PIL.ImageTk.PhotoImage(image=image2)
    #         self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
    #
    #     self.window.after(self.delay, self.update)

    def update2(self, indexs):
        if indexs[0] == -1 and indexs[1] == -1:
            self.ret1, self.frame1 = self.cap1.read()
            self.ret2, self.frame2 = self.cap2.read()
        else:
            self.ret1, self.frame1, self.frame_org1 = self.vid.get_frame()
            self.ret2, self.frame2, self.frame_org2 = self.vid2.get_frame()

        if self.rotateVal.get() == 2:
            self.frame1 = cv2.rotate(self.frame1, cv2.ROTATE_90_CLOCKWISE)
            self.frame2 = cv2.rotate(self.frame2, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotateVal.get() == 3:
            self.frame1 = cv2.rotate(self.frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.frame2 = cv2.rotate(self.frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.reverseValLR.get() != 1:
            self.frame1 = cv2.flip(self.frame1, 1)  # 좌우반전
            self.frame2 = cv2.flip(self.frame2, 1)  # 좌우반전

        if self.reverseValUD.get() != 1:
            self.frame1 = cv2.flip(self.frame1, 0)  # 상하반전
            self.frame2 = cv2.flip(self.frame2, 0)  # 상하반전

        current_time = time.time() - self.prev_time  # 경과시간 = 현재시간 - 이전 프레임 재생시간
        # print(current_time)

        # print("ret1")
        # print(self.ret1)
        # print("ret2")
        # print(self.ret2)

        if self.ret1 and self.ret2:
            if current_time > 1. / self.fps:  # fps 최대 제한
                pass
            else:
                while (True):
                    current_time = time.time() - self.prev_time
                    if current_time > 1. / self.fps:
                        break

            if self.is_recording:
                self.frame_recode1 = self.frame_org1
                self.frame_recode2 = self.frame_org2
                if self.rotateVal.get() != 1:
                    self.frame_recode1 = self.rotate(self.frame_recode1)
                    self.frame_recode2 = self.rotate(self.frame_recode2)
                if self.reverseValLR.get() != 1 or self.reverseValUD.get() != 1:
                    self.frame_recode1 = self.reverse(self.frame_recode1)
                    self.frame_recode2 = self.reverse(self.frame_recode2)
                self.writer.write(self.frame_recode1)
                self.writer2.write(self.frame_recode2)

            self.prev_time = time.time()

            self.block1Frame1 = self.frame1.copy()
            self.block1Frame2 = self.frame2.copy()
            self.block2Frame1 = self.frame1.copy()
            self.block2Frame2 = self.frame2.copy()


            # self.blockTempFrame1 = self.tempFrame1.copy()
            # self.blockTempFrame2 = self.tempFrame2.copy()

            # tempframe1 = cv2.resize(self.block1Frame1, (261, 464))
            # tempframe2 = cv2.resize(self.block1Frame2, (261, 464))
            #
            # image1 = PIL.Image.fromarray(cv2.cvtColor(tempframe1, cv2.COLOR_BGR2RGB))
            # self.photo1 = PIL.ImageTk.PhotoImage(image=image1)
            # self.frontVideoIma1 = self.frontVideo.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)
            #
            # image2 = PIL.Image.fromarray(cv2.cvtColor(tempframe2, cv2.COLOR_BGR2RGB))
            # self.photo2 = PIL.ImageTk.PhotoImage(image=image2)
            # self.backVideoIma2 = self.backVideo.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

            ### block1 START ###
            drawBlockImage1 = Tracking_Extract_and_detect_color1.frontCamFunc(self.block1Frame1)
            # drawBlockImage1 = Tracking_Extract_and_detect_color1.frontCamFunc(self.blockTempFrame1)
            image1 = PIL.Image.fromarray(cv2.cvtColor(drawBlockImage1, cv2.COLOR_BGR2RGB))
            self.photo1 = PIL.ImageTk.PhotoImage(image=image1)
            self.frontVideoIma1 = self.frontVideo.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)

            drawBlockImage2, self.prevTime = Tracking_Extract_and_detect_color1.middleCamFunc(self.block1Frame2, self.prevTime)
            # drawBlockImage2 = Tracking_Extract_and_detect_color1.middleCamFunc(self.blockTempFrame2)
            image2 = PIL.Image.fromarray(cv2.cvtColor(drawBlockImage2, cv2.COLOR_BGR2RGB))
            self.photo2 = PIL.ImageTk.PhotoImage(image=image2)
            self.backVideoIma2 = self.backVideo.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

            if self.histBool1:
                gaussianBlurImage1 = cv2.GaussianBlur(self.block1Frame1, (0, 0), 1)
                # gaussianBlurImage1 = cv2.GaussianBlur(self.blockTempFrame1, (0, 0), 1)

                normalizationFrame1 = cv2.normalize(gaussianBlurImage1, None, 0, 255, cv2.NORM_MINMAX)

                channels1 = cv2.split(normalizationFrame1)

                self.histogramGraphAxs[0].clear()

                for (ch, color) in zip(channels1, self.colors):
                    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
                    self.histogramGraphAxs[0].plot(hist, color=color)


                hist = cv2.calcHist([gaussianBlurImage1], [0], None, [256], [0, 256])
                self.histogramGraphAxs[0].plot(hist, 'k')

                self.histogramGraphFigure.canvas.draw()
                self.histogramGraphFigure.canvas.flush_events()

                histogramImage = PIL.Image.fromarray(np.array(self.histogramGraphFigure.canvas.renderer._renderer))
                histogramImage.resize((270, 960))
                self.histogramPhoto = PIL.ImageTk.PhotoImage(image=histogramImage)
                self.histogramGraph.create_image(0, 0, image=self.histogramPhoto, anchor=tkinter.NW)

                self.histBool1 = False

            if self.histBool2:
                gaussianBlurImage2 = cv2.GaussianBlur(self.block1Frame2, (0, 0), 1)
                # gaussianBlurImage2 = cv2.GaussianBlur(self.blockTempFrame2, (0, 0), 1)

                normalizationFrame2 = cv2.normalize(gaussianBlurImage2, None, 0, 255, cv2.NORM_MINMAX)

                channels2 = cv2.split(normalizationFrame2)

                self.histogramGraphAxs[1].clear()

                for (ch, color) in zip(channels2, self.colors):
                    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
                    self.histogramGraphAxs[1].plot(hist, color=color)

                hist = cv2.calcHist([gaussianBlurImage2], [0], None, [256], [0, 256])
                self.histogramGraphAxs[1].plot(hist, 'k')

                self.histogramGraphFigure.canvas.draw()
                self.histogramGraphFigure.canvas.flush_events()

                histogramImage = PIL.Image.fromarray(np.array(self.histogramGraphFigure.canvas.renderer._renderer))
                histogramImage.resize((270, 960))
                self.histogramPhoto = PIL.ImageTk.PhotoImage(image=histogramImage)
                self.histogramGraph.create_image(0, 0, image=self.histogramPhoto, anchor=tkinter.NW)

                self.histBool2 = False

            ### block1 END ###

            ### block2 START ###
            LaneRecovery.model_gen.eval()
            # Timeserise_transformer_v6_Run.model.eval()

            basicImage1 = PIL.Image.fromarray(cv2.cvtColor(self.block2Frame1, cv2.COLOR_BGR2RGB))
            # basicImage1 = PIL.Image.fromarray(cv2.cvtColor(self.blockTempFrame1, cv2.COLOR_BGR2RGB))
            basicImage1 = LaneRecovery.transform(basicImage1)
            basicImage1 = basicImage1.unsqueeze(0)
            fake_imgs1 = LaneRecovery.model_gen(basicImage1.to(LaneRecovery.device)).detach().cpu()
            fake_imgs1 = fake_imgs1.squeeze()
            # demormalize
            fake_imgs1 = LaneRecovery.to_pil_image(0.5 * fake_imgs1 + 0.5)
            numpy_image1 = np.array(fake_imgs1)
            fake_imgs1 = cv2.resize(numpy_image1, (720, 1280))

            dis1, centerValue1, self.shiftArray1, drawBlockResult1 = Tracking_Extract_and_detect_color2.frontCamFunc(fake_imgs1, self.shiftArray1)
            self.basicPhoto1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(dis1, cv2.COLOR_BGR2RGB)))
            self.basicPhotoIma1 = self.frontRecoveryVideo.create_image(0, 0, image=self.basicPhoto1, anchor=tkinter.NW)

            basicImage2 = PIL.Image.fromarray(cv2.cvtColor(self.block2Frame2, cv2.COLOR_BGR2RGB))
            # basicImage2 = PIL.Image.fromarray(cv2.cvtColor(self.blockTempFrame2, cv2.COLOR_BGR2RGB))
            basicImage2 = LaneRecovery.transform(basicImage2)
            basicImage2 = basicImage2.unsqueeze(0)
            fake_imgs2 = LaneRecovery.model_gen(basicImage2.to(LaneRecovery.device)).detach().cpu()
            fake_imgs2 = fake_imgs2.squeeze()
            # demormalize
            fake_imgs2 = LaneRecovery.to_pil_image(0.5 * fake_imgs2 + 0.5)
            numpy_image2 = np.array(fake_imgs2)
            fake_imgs2 = cv2.resize(numpy_image2, (720, 1280))

            dis2, centerValue2, self.shiftArray2, drawBlockResult2, self.laneTrackingBool, self.nozzleState = Tracking_Extract_and_detect_color2.middleCamFunc(fake_imgs2, self.shiftArray2, self.laneTrackingBool)
            self.basicPhoto2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(dis2, cv2.COLOR_BGR2RGB)))
            self.basicPhotoIma2 = self.backRecoveryVideo.create_image(0, 0, image=self.basicPhoto2, anchor=tkinter.NW)

            self.animate(centerValue1, centerValue2, drawBlockResult1, drawBlockResult2)

            # 노즐OnOff
            if hasattr(self, 'thread'):
                if self.thread.seq.isOpen():
                    if self.nozzleState != self.beforeNozzleState:
                        msg = "SV1" if self.nozzleState == True else "SV0"
                        self.thread.seq.write(bytes(msg + "\\n", encoding="ascii"))

                    self.beforeNozzleState = self.nozzleState
            ### block2 END ###

        if self.start == True:
            self.window.after(self.delay, lambda: self.update2(indexs))

    def getGrayHistImage(self, hist):
        # 가장 높은 높이가 100으로 제한을 둠
        imgHist = np.full((280, 320), 0, dtype=np.uint8)  # 100=>200, 256=>280

        histMax = np.max(hist)  # histmax = 255
        for x in range(320):
            pt1 = (x, 280)  # 시작점, 좌측 상단 기준
            pt2 = (x, 280 - int(hist[x, 0] * 280 / histMax))  # 끝점, 100을 곱하고 255로 나눠 단위 통일
            cv2.line(imgHist, pt1, pt2, 255)  # 직선을 그려 히스토그램 그리기

        return imgHist

    def rotate(self, frame):
        if self.rotateVal.get() == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 시계방향
        elif self.rotateVal.get() == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 반시계방향
        else:
            pass
        return frame

    def reverse(self, frame):
        if self.reverseValLR.get() == 2:
            frame = cv2.flip(frame, 1)  # 좌우반전
        if self.reverseValUD.get() == 2:
            frame = cv2.flip(frame, 0)  # 상하반전
        return frame

    # def histogram(self, frame):
    #     grayscale_frame = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    #     colort_frame = self.frame_org
    #     grayscale_his1 = cv2.calcHist([grayscale_frame], [0], None, [256], [0, 256])
    #     grayscale_his2 = np.histogram(grayscale_his1.ravel(), 256, [0, 256])

    def __del__(self):
        if self.is_recording:
            self.vid.video.release()


class MyVideoCapture:
    def __init__(self, video_source, width, height, resize, option):
        self.video_source = video_source
        if option == None:
            print("no option")
            self.webcam = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        else:
            print("yes option")
            print(option)
            self.webcam = cv2.VideoCapture(video_source + option, cv2.CAP_DSHOW)
        self.width = width
        self.height = height
        self.resize = resize
        if not self.webcam.isOpened():
            print("not open")
            raise ValueError("Unable to open video source", video_source)

        self.fourcc = cv2.VideoWriter_fourcc(*'DX50')
        # self.width = self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.webcam.set(cv2.CAP_PROP_SETTINGS, 1)

        # self.prevTime = 0

    def get_fourcc(self):
        return self.fourcc

    def get_webcam(self):
        return self.webcam

    def create_video(self, fps):
        file_name = str(datetime.datetime.now()) + '.mp4'
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # opencv 3.0
        # Error: 'module' object has no attribute 'VideoWriter_fourcc'
        # fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        # jpeg,h263,'m', 'p', '4', 'v'
        self.video = cv2.VideoWriter(file_name, fourcc, fps / 3.0, (self.width, self.height))

    def get_frame(self):
        def draw_text(img, text, x, y, text_color, font_size):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            # text_color = (255, 0, 0)
            text_color_bg = (0, 0, 0)

            text_size, _ = cv2.getTextSize(text, font, font_size, font_thickness)
            text_w, text_h = text_size
            offset = 5
            # cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
            # cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_size, text_color, font_thickness)

        if self.webcam.isOpened():
            ret, frame_org = self.webcam.read()


            if ret:
                # frame_org = cv2.flip(frame_org, 1)  # 반전
                # frame_org = cv2.rotate(frame_org, cv2.ROTATE_90_CLOCKWISE)
                # frame_resize = cv2.resize(frame_org, dsize=(0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)
                frame_resize = cv2.resize(frame_org, dsize=(270, 480), interpolation=cv2.INTER_LINEAR)

                # frame = cv2.flip(frame_resize, 1)

                image_title = f'Camera{self.video_source}'
                draw_text(frame_resize, image_title, 30, 10, (255, 0, 255), 0.7)

                return (ret, frame_resize, frame_org)
            else:
                return (ret, None, None)
        else:
            return (False, None, None)

    def __del__(self):
        if self.webcam.isOpened():
            self.webcam.release()


class MainClass:
    def __init__(self):
        self.indexs = []
        self.linewin = tkinter.Tk()
        self.dev_arr = []
        self.cnnState = 'normal'
        self.clss = None
        self.clickVal = None
        self.index1 = None
        self.index2 = None
        self.option = None

    def create_menu(self):
        self.menu = tkinter.Menu(self.linewin)
        self.menuCamera = tkinter.Menu(self.menu, tearoff=0)
        self.menuCamera.add_command(label="연결", command=self.cnnCamera, state='normal')
        # self.menuCamera.add_command(label="연결종료", command=self.disCnnCamera, state='normal')
        self.menu.add_cascade(label="카메라", menu=self.menuCamera)
        self.linewin.config(menu=self.menu)

    def reset_camera_index(self):
        self.Camera1Val.configure(text="")
        self.Camera2Val.configure(text="")
        self.index1 = None
        self.index2 = None

    def find_dev(self):
        self.reset_camera_index()
        self.table.delete(*self.table.get_children())
        index = 0
        arr = []
        while True:
            frontVideo = cv2.VideoCapture(index + cv2.CAP_DSHOW)
            if not frontVideo.read()[0]:
                break
            else:
                arr.append(index)
                self.table.insert('', 'end', values=(index, index))
            frontVideo.release()
            index += 1

        self.dev_arr = arr

    def main(self):
        # self.create_menu()
        # self.find_dev()
        self.linewin.title("Tkinter and OpenCV")
        self.linewin.geometry("1920x1080")
        self.linewin.configure(bg='black')
        # self.linewin.attributes("-fullscreen", True)
        self.linewin.resizable(False, True)
        self.linewin.state("zoomed")
        self.clss = App(self, self.linewin)
        self.linewin.protocol('WM_DELETE_WINDOW', self.exitWindow)
        self.linewin.mainloop()

        del self.linewin


    def exitWindow(self):
        self.clss.txtFile.close()

        self.clss.close_port()
        self.linewin.destroy()
        self.linewin.quit()


    def tableClick(self, event):
        selectedItem = self.table.focus()
        getValue = self.table.item(selectedItem).get('values')  # 딕셔너리의 값만 가져오기
        self.clickVal = getValue[0]  # 클릭한 값

    def setting1(self):
        self.Camera1Val.configure(text=self.clickVal)
        self.index1 = int(self.clickVal)
        if self.Camera2Val.cget("text") == self.clickVal:
            self.index2 = None
            self.Camera2Val.configure(text="")

    def setting2(self):
        self.Camera2Val.configure(text=self.clickVal)
        self.index2 = int(self.clickVal)
        if self.Camera1Val.cget("text") == self.clickVal:
            self.index1 = None
            self.Camera1Val.configure(text="")

    def cnnWindowClosing(self):
        if messagebox.askokcancel("Quit", "창을 닫으시겠습니까?"):
            self.disCnnCamera()
            self.cnnWindow.destroy()
            self.clss.cameraConnectBtn["state"] = tkinter.NORMAL
            # self.menuCamera.delete(0, 0)
            # self.menuCamera.add_command(label="연결", command=self.cnnCamera, state='normal')
            # self.soc
            # ket.close()

    def close_cameraSetting(self):
        if messagebox.askokcancel("Quit", "창을 닫으시겠습니까?"):
            self.cameraSettingModal.destroy()

    # def open_cameraSetting(self):
    #     self.cameraSettingModal = tkinter.Toplevel(self.linewin)
    #     self.cameraSettingModal.protocol("WM_DELETE_WINDOW", self.close_cameraSetting)
    #
    #     self.cameraSettingBar1 = tkinter.Frame(self.cameraSettingModal, width=320, height=240)
    #     self.cameraSettingBar1.grid(column=0, row=1, padx=0, pady=0, sticky="nsew")
    #     self.cameraSettingBar2 = tkinter.Frame(self.cameraSettingModal, width=320, height=240)
    #     self.cameraSettingBar2.grid(column=1, row=1, padx=0, pady=0, sticky="nsew")
    #
    #     self.clss.create_trackbar()

    def cnnCamera(self):
        self.clss.cameraConnectBtn["state"] = tkinter.DISABLED
        # self.menuCamera.delete(0, 0)
        # self.menuCamera.add_command(label="연결", command=self.cnnCamera, state='disabled')
        self.index1 = None
        self.index2 = None

        self.cnnWindow = tkinter.Toplevel(self.linewin)
        self.cnnWindow.protocol("WM_DELETE_WINDOW", self.cnnWindowClosing)
        self.frm0 = tkinter.Frame(self.cnnWindow, relief="solid", bd=1, )
        self.frm0.pack(expand=True, fill="both")
        self.frm1 = tkinter.Frame(self.cnnWindow, relief="solid", bd=1, )
        self.frm1.pack(expand=True, fill="both")
        self.frm2 = tkinter.Frame(self.cnnWindow, relief="solid", bd=1, )
        self.frm2.pack(expand=True, fill="both")
        self.frm3 = tkinter.Frame(self.cnnWindow, relief="solid", bd=1, )
        self.frm3.pack(expand=True, fill="both")

        self.optionBtn = tkinter.Button(self.frm0, text='옵션선택', width=10, command=self.setOption)
        self.optionLbl = tkinter.Label(self.frm0, text='None')
        self.fpsLbl = tkinter.Label(self.frm0, text='FPS 입력')
        self.inputFps = tkinter.Entry(self.frm0)
        self.inputFps.insert(0, "13")
        self.setCamera1 = tkinter.Button(self.frm1, text='카메라1 선택', width=10, command=self.setting1)
        self.setCamera2 = tkinter.Button(self.frm1, text='카메라2 선택', width=10, command=self.setting2)
        self.Camera1Val = tkinter.Label(self.frm1)
        self.Camera2Val = tkinter.Label(self.frm1)
        self.confirm = tkinter.Button(self.frm3, text='시작', width=10, command=self.startCnn)
        self.cnnStop = tkinter.Button(self.frm3, text='종료', width=10, command=self.disCnnCamera, state=tkinter.DISABLED)
        self.findCameraIdx = tkinter.Button(self.frm3, text='탐색', width=10, command=self.find_dev)
        self.optionBtn.grid(column=0, row=0)
        self.optionLbl.grid(column=1, row=0)
        self.fpsLbl.grid(column=0, row=1)
        self.inputFps.grid(column=1, row=1)
        self.setCamera1.grid(column=1, row=0)
        self.setCamera2.grid(column=1, row=1)
        self.Camera1Val.grid(column=2, row=0)
        self.Camera2Val.grid(column=2, row=1)
        self.confirm.grid(column=0, row=0)
        self.cnnStop.grid(column=1, row=0)
        self.findCameraIdx.grid(column=3, row=0)

        self.table = ttk.Treeview(self.frm2, columns=(1), height=11, show="headings")
        self.table.pack(expand=True, fill="both")
        self.table.heading(1, text="number")
        self.table.bind('<ButtonRelease-1>', self.tableClick)  # 테이블 클릭한 값 받기

        self.find_dev()

    def setOption(self):
        if self.option == "CAP_DSHOW":
            self.optionLbl.config(text="None")
            self.option = None
        else:
            self.optionLbl.config(text="CAP_DSHOW")
            self.option = "CAP_DSHOW"

    def startCnn(self):
        if self.index1 == None and self.index2 == None:
            # messagebox.askokcancel("경고", "연결할 카메라를 세팅해주세요.")
            self.confirm["state"] = tkinter.DISABLED
            self.findCameraIdx["state"] = tkinter.DISABLED
            self.cnnStop["state"] = tkinter.NORMAL

            indexs = [-1, -1]
            fps = self.inputFps.get()
            self.clss.startFilming(indexs, self.option, fps)
        else:
            self.confirm["state"] = tkinter.DISABLED
            self.findCameraIdx["state"] = tkinter.DISABLED
            self.cnnStop["state"] = tkinter.NORMAL

            if self.index1 == None:
                indexs = [self.index2]
            elif (self.index2 == None):
                indexs = [self.index1]
            else:
                indexs = [self.index1, self.index2]
            fps = self.inputFps.get()
            self.clss.startFilming(indexs, self.option, fps)
            # self.cnnWindow.destroy()

    def disCnnCamera(self):
        self.confirm["state"] = tkinter.NORMAL
        self.findCameraIdx["state"] = tkinter.NORMAL
        self.cnnStop["state"] = tkinter.DISABLED
        self.clss.stopFilming()

# SerialThread.is_run = False
c = MainClass()
c.main()
