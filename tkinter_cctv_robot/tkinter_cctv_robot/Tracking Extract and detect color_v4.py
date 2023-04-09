import cv2
import numpy as np
import os

#데이터 저장 =====================
# os.chdir('D:/New Project/2. 유일로보틱스 과제/차선 촬영 영상/') # 디렉토리
os.chdir(rf'C:\Users\TECH\Desktop\yuil_1108\yuil\mv') # 디렉토리


ff = open("Front_Tracking_01.csv", 'w') # 전방차선 추출 자료 저장 파일 이름
ff.write('Frame, Shift'+'\n')

fc = open("Center_Tracking_01.csv", 'w') #중앙차선 추출 자료 저장 파일 이름
fc.write('Frame, Shift'+'\n')

#데이터 추출 Parameters =====================
#전방 =======
cap_f = cv2.VideoCapture(rf'C:\Users\TECH\Desktop\yuil_1108\yuil\mv\22.10.13_f_mv_1\camera1_2022-10-13_12-49-14.avi') #전방 차선 영상 파일
width = cap_f.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap_f.get(cv2.CAP_PROP_FPS)
count = 0
pcent1, pcent2, pcent3, pcent4, pcent5 = [0,0,0,0,0]
state1, state2, state3, state4, state5 = [False, False, False, False, False]
y_block1, y_block2, y_block3, y_block4, y_block5 = [5, 15, 35, 60, 100]
track1, track2, track3, track4, track5 = [0,0,0,0,0]
track = 0
start = 0
tx = 120

#중앙 =======
cap_c = cv2.VideoCapture(rf'C:\Users\TECH\Desktop\yuil_1108\yuil\mv\22.10.13_b_mv_1\camera2_2022-10-13_12-49-14.avi') #후방 차선 영상 파일
width2 = cap_c.get(cv2.CAP_PROP_FRAME_WIDTH)
height2 = cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps2 = cap_c.get(cv2.CAP_PROP_FPS)
pcentc1, pcentc2 = [0,0]
statec1, statec2 = [False, False]
y_blockc1, y_blockc2, y_blockc3 = [35, 57, 100] 
statec1, statec2 = [False, False]
trackc = 0
startc = 0
nozzle_onoff = 0 #초기 노즐 상태 변경(없음)
nozzle_state = 0 #초기 노즐 상태(닫힘)
disp = 10
textcc = ""
missed = 0

# 탐지 및 추적, 확인 Processing =====================================
if cap_f.isOpened() and cap_c.isOpened():
    while True:
        ret, frame = cap_f.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.resize(frame, (0,0), fx = 0.6, fy = 0.6, interpolation=cv2.INTER_AREA)

#중앙 비디오 =====================================            
        ret2, frame2 = cap_c.read()
        if ret2:
            frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame2 = cv2.resize(frame2, (0,0), fx = 0.6, fy = 0.6, interpolation=cv2.INTER_AREA)

            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            hi, wi, _ = frame.shape
            hi2, wi2, _ = frame2.shape
            
            roi = int(wi/5)
            roi2 = int(wi2/30)
            
            lines1, lines2, lines3, line4, line5 = [None, None, None, None, None]
            line = cv2.line(frame, (int(wi/2), 0), (int(wi/2), hi), (255,0,0), 2, cv2.LINE_AA)
            lien = cv2.line(frame2, (int(wi2/2), 0), (int(wi2/2), hi), (255,0,0), 2, cv2.LINE_AA)
            
            line = cv2.line(frame, (int(wi/2) - 50, 0), (int(wi/2) + 50, 0), (255, 0, 0), 3, cv2.LINE_AA)
            line = cv2.line(frame, (int(wi/2) - 50, int((hi*y_block1)/100)), (int(wi/2) + 50, int((hi*y_block1)/100)), (255, 0, 0), 3, cv2.LINE_AA)
            line = cv2.line(frame, (int(wi/2) - 50, int((hi*y_block2)/100)), (int(wi/2) + 50, int((hi*y_block2)/100)), (255, 0, 0), 3, cv2.LINE_AA)
            line = cv2.line(frame, (int(wi/2) - 50, int((hi*y_block3)/100)), (int(wi/2) + 50, int((hi*y_block3)/100)), (255, 0, 0), 3, cv2.LINE_AA)
            line = cv2.line(frame, (int(wi/2) - 50, int((hi*y_block4)/100)), (int(wi/2) + 50, int((hi*y_block4)/100)), (255, 0, 0), 3, cv2.LINE_AA)
            cent1, cent2, cent3, cent4, cent5 = [None, None, None, None, None]
            
 
            remain = 0
            line = cv2.line(frame2, (int(wi2/2) - 50, int((hi2*y_blockc1)/100)), (int(wi2/2) + 50, int((hi2*y_blockc1)/100)), (255, 0, 0), 3, cv2.LINE_AA)            
            line = cv2.line(frame2, (int(wi2/2) - 50, int((hi2*y_blockc2)/100)), (int(wi2/2) + 50, int((hi2*y_blockc2)/100)), (255, 0, 0), 3, cv2.LINE_AA)            
            line = cv2.line(frame2, (int(wi2/2) - 50, int((hi2*y_blockc3)/100)), (int(wi2/2) + 50, int((hi2*y_blockc3)/100)), (255, 0, 0), 3, cv2.LINE_AA)
            start_end1, start_end2 = [False, False]
            centc1, centc2, abcent = [None, None, None]

# 중앙 Guide_1(Blockc1) ==================================================
            blockc1 = gray2[int((hi2*y_blockc1)/100):int((hi2*y_blockc2)/100), roi2: wi2-roi2]
            blockc1 = cv2.bilateralFilter(blockc1, -1, 100, 10)
##            sigma = 1
##            block1 = cv2.GaussianBlur(block1, (0,0), sigma)
            retc1, thrc1 = cv2.threshold(blockc1, 100, 255, cv2.THRESH_OTSU)
            contoursc1, _ = cv2.findContours(thrc1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contc1 in contoursc1:
                approxc1 = cv2.approxPolyDP(contc1, cv2.arcLength(contc1, True) * 0.02, True)
                if len(approxc1) == 4:
                    (x, y, w, h) = cv2.boundingRect(contc1)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(blockc1, contoursc1, -1, (255,255,255), 4)
#                    print(w)
                    if ((w > 110) and (w < 145)):
                        pt1 = (x+roi2+int(w/2)-2, y+ int((hi2*y_blockc1)/100))
                        pt2 = (x + roi2+ int(w/2)+2, y + int((hi2*y_blockc1)/100)+h)
                        cv2.rectangle(frame2, pt1, pt2, (0,0,255),-1)
                        if (h > (blockc1.shape[0]/20)) and (h < int(blockc1.shape[0]-(blockc1.shape[0]/20))):
                            start_end1 = True
                            nozzle1 = (y, h)
                        centc1 = int(pt1[0] + 2 - int(wi2/2))

# 중앙 Guide_2(Blockc2) ==================================================
            blockc2 = gray2[int((hi2*y_blockc2)/100):int((hi2*y_blockc3)/100), roi2: wi2-roi2]
            blockc2 = cv2.bilateralFilter(blockc2, -1, 100, 10)
##            sigma = 1
##            block1 = cv2.GaussianBlur(block1, (0,0), sigma)
            retc2, thrc2 = cv2.threshold(blockc2, 100, 255, cv2.THRESH_OTSU)
            contoursc2, _ = cv2.findContours(thrc2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contc2 in contoursc2:
                approxc2 = cv2.approxPolyDP(contc2, cv2.arcLength(contc2, True) * 0.02, True)
                if len(approxc2) == 4:
                    (x, y, w, h) = cv2.boundingRect(contc2)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(blockc2, contoursc2, -1, (255,255,255), 4)
#                    print(w)
                    if ((w > 130) and (w < 205)):
                        pt1 = (x+roi2+int(w/2)-2, y+ int((hi2*y_blockc2)/100))
                        pt2 = (x + roi2+ int(w/2)+2, y + int((hi2*y_blockc2)/100)+h)
                        cv2.rectangle(frame2, pt1, pt2, (0,0,255),-1)
                        if (h > (blockc2.shape[0]/30)) and (h < int(blockc2.shape[0]-(blockc2.shape[0]/30))):
                            start_end2 = True
                            nozzle2 = (y, h)
                        centc2 = int(pt1[0] + 2 - int(wi2/2))

# 중앙 Absolute Position(Blockc3) ==================================================
            blockc3 = gray2[(hi2-200):(hi2), roi2: wi2-roi2]
            blockc3 = cv2.bilateralFilter(blockc3, -1, 120, 10)
            retc3, thrc3 = cv2.threshold(blockc3, 100, 255, cv2.THRESH_OTSU)
            contoursc3, _ = cv2.findContours(thrc3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contc3 in contoursc3:
                approxc3 = cv2.approxPolyDP(contc3, cv2.arcLength(contc3, True) * 0.02, True)
                if len(approxc3) == 4:
                    (x, y, w, h) = cv2.boundingRect(contc3)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(blockc3, contoursc3, -1, (255,255,255), 4)
#                    print(w)
                    if ((w > 170) and (w < 210)):
                        pt1 = (x+roi2+int(w/2)-10, y+ (hi2-110))
                        pt2 = (x + roi2+ int(w/2)+10, y + (hi2-90))
                        cv2.rectangle(frame2, pt1, pt2, (0,255,0),-1)
                        abcent = int(pt1[0] + 10)

# 중앙 차선 추종 =================================================
            if (centc2 != None) and (pcentc2 != None):
                if pcentc2 == 0:
                    pcentc2 = centc2
                trackc = trackc + (centc2 - pcentc2)
                pcentc2 = centc2
                pcentc1 = centc1
            else:
                if (centc1 != None) and (pcentc1 != None):
                    if pcentc1 == 0:
                        pcentc1 = centc1
                    trackc = trackc + int((centc1 - pcentc1))
                    pcentc1 = centc1
                else:
                    trackc = trackc + missed ###########(sync 필요)
                
# 중앙 노즐 제어  =================================================
# 거시적 제어 (On/Off)
            if (centc2 != None) or (centc1 != None):
                nozzle_state = 1 #노즐 열기
            else:
                nozzle_state = 0 #노즐 닫기


# 상세 제어(1/100 초) ==================================================
            if start_end1 or start_end2:
                if start_end2:
                    if nozzle2[0] > 0:
                        nozzle_onoff = 0 #노즐 닫기
                        remain = 10 - int((nozzle2[1]/blockc2.shape[0])*10)
#                        print('노즐 닫기', remain)
                    else:
                        nozzle_onoff = 1 #노즐 열기
                        remain = 10 - int((nozzle2[1]/blockc2.shape[0])*10)
#                        print('노즐 열기', remain)
                elif start_end1:
                    if nozzle1[0] > 0:
                        nozzle_onoff = 0 #노즐 닫기
                        remain = 20 - int((nozzle1[1]/blockc1.shape[0])*10)
#                        print('노즐 닫기', remain)
                    else:
                        nozzle_onoff = 1 #노즐 열기
                        remain = 20 - int((nozzle1[1]/blockc1.shape[0])*10)
#                        print('노즐 열기', remain)
                        
                        

# 전방 Guide_1(Block1) ==================================================
            block1 = gray[0:int((hi*y_block1)/100), roi: wi-roi]
            sigma = 1
            block1 = cv2.GaussianBlur(block1, (0,0), sigma)
            ret1, thr1 = cv2.threshold(block1, 100, 255, cv2.THRESH_OTSU)
            contours1, _ = cv2.findContours(thr1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont1 in contours1:
                approx1 = cv2.approxPolyDP(cont1, cv2.arcLength(cont1, True) * 0.02, True)
                if len(approx1) == 4:
                    (x, y, w, h) = cv2.boundingRect(cont1)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(block1, contours1, -1, (255,255,255), 4)
#                    print(w)
                    if ((w > 10) and (w < 15)):
                        pt1 = (x+roi + int(w/2)-2, y)
                        pt2 = (x + roi+ int(w/2)+2, y +h)
                        cv2.rectangle(frame, pt1, pt2, (0,0,255),-1)
                        cent1 = int(pt1[0] + 2 - int(wi/2))
                        
# 전방 Guide_2(Block2) ==================================================
            block2 = gray[int((hi*y_block1)/100):int((hi*y_block2)/100), roi: wi-roi]
            sigma = 2
            block2 = cv2.GaussianBlur(block2, (0,0), sigma)
            ret2, thr2 = cv2.threshold(block2, 180, 255, cv2.THRESH_OTSU) #Mask를 할 경우 삭제
            contours2, _ = cv2.findContours(thr2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont2 in contours2:
                approx2 = cv2.approxPolyDP(cont2, cv2.arcLength(cont2, True) * 0.02, True)
                if len(approx2) == 4:
                    (x, y, w, h) = cv2.boundingRect(cont2)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(block2, contours2, -1, (255,255,255), 4)
##                    print(w)
                    if ((w > 15) and (w < 25)):
                        pt1 = (x+roi+int(w/2)-2, y+ int((hi*y_block1)/100))
                        pt2 = (x + roi+ int(w/2)+2, y + int((hi*y_block1)/100)+h)
                        cv2.rectangle(frame, pt1, pt2, (0,0,255),-1)
                        cent2 = pt1[0] + 2 - int(wi/2)

# 전방 Guide_3(Block3) ==================================================
            block3 = gray[int((hi*y_block2)/100):int((hi*y_block3)/100), roi: wi-roi]
            sigma = 3
            block3 = cv2.GaussianBlur(block3, (0,0), sigma)
            ret3, thr3 = cv2.threshold(block3, 180, 255, cv2.THRESH_OTSU)
            contours3, _ = cv2.findContours(thr3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont3 in contours3:
                approx3 = cv2.approxPolyDP(cont3, cv2.arcLength(cont3, True) * 0.02, True)
                if len(approx3) == 4:
                    (x, y, w, h) = cv2.boundingRect(cont3)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(block3, contours3, -1, (255,255,255), 4)
#                    print(w)
                    if ((w > 25) and (w < 45)):
                        pt1 = (x+roi+int(w/2)-2, y+ int((hi*y_block2)/100))
                        pt2 = (x + roi+ int(w/2)+2, y + int((hi*y_block2)/100) + h)
                        cv2.rectangle(frame, pt1, pt2, (0,0,255),-1)
                        cent3 = pt1[0] + 2 - int(wi/2)

# 전방 Guide_4(Block4) ==================================================
            block4 = gray[int((hi*y_block3)/100):int((hi*y_block4)/100), roi: wi-roi]
            sigma = 4
            block4 = cv2.GaussianBlur(block4, (0,0), sigma)
            ret4, thr4 = cv2.threshold(block4, 180, 255, cv2.THRESH_OTSU)
            contours4, _ = cv2.findContours(thr4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont4 in contours4:
                approx4 = cv2.approxPolyDP(cont4, cv2.arcLength(cont4, True) * 0.02, True)
                if len(approx4) == 4:
                    (x, y, w, h) = cv2.boundingRect(cont4)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(block4, contours4, -1, (255,255,255), 4)
##                    # 차선 검출 조건 지정(하단)
#                    print(w)
                    if ((w > 40) and (w < 65)):
                        pt1 = (x+roi+int(w/2)-2, y+ int((hi*y_block3)/100))
                        pt2 = (x + roi+ int(w/2)+2, y + int((hi*y_block3)/100) + h)
                        cv2.rectangle(frame, pt1, pt2, (0,0,255),-1)
                        cent4 = pt1[0] + 2 - int(wi/2)

# 전방 Guide_5(Block5) ==================================================
            block5 = gray[int((hi*y_block4)/100):int((hi*y_block5)/100), roi: wi-roi]
            sigma = 4
            block5 = cv2.GaussianBlur(block5, (0,0), sigma)
            ret5, thr5 = cv2.threshold(block5, 180, 255, cv2.THRESH_OTSU)
            contours5, _ = cv2.findContours(thr5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cont5 in contours5:
                approx5 = cv2.approxPolyDP(cont5, cv2.arcLength(cont5, True) * 0.02, True)
                if len(approx5) == 4:
                    (x, y, w, h) = cv2.boundingRect(cont5)
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)
                    cv2.drawContours(block5, contours5, -1, (255,255,255), 4)
##                    # 차선 검출 조건 지정(하단)
##                    print(w)
                    if ((w > 80) and (w < 105)):
                        pt1 = (x+roi+int(w/2)-2, y+ int((hi*y_block4)/100))
                        pt2 = (x + roi+ int(w/2)+2, y + int((hi*y_block4)/100) + h)
                        cv2.rectangle(frame, pt1, pt2, (0,0,255),-1)
                        cent5 = pt1[0] + 2 - int(wi/2)

# 전방 데이터 추출 및 앵커 운영, 선택 ========================================
# state, track =======
   
            if (cent1 != None) and (pcent1 !=0) and (pcent1 !=None):
                if abs(cent1 - pcent1) < 20:
                    track1 = cent1 - pcent1
                    state1 = True
                else:
                    cent1 = pcent1
            else:
                state1 = False

            if (cent2 != None) and (pcent2 !=0) and (pcent2 !=None):
                if abs(cent2 - pcent2) < 20:
                    track2 = cent2 - pcent2
                    state2 = True
                else:
                    cent2 = pcent2
            else:
                state2 = False

            if (cent3 != None) and (pcent3 !=0) and (pcent3 !=None):
                if abs(cent3 - pcent3) < 20:
                    track3 = cent3 - pcent3
                    state3 = True
                else:
                    cent3 = pcent3
            else:
                state3 = False

            if (cent4 != None) and (pcent4 !=0) and (pcent4 !=None):
                if abs(cent4 - pcent4) < 20:
                    track4 = cent4 - pcent4
                    state4 = True
                else:
                    cent4 = pcent4
            else:
                state4 = False
        
            if (cent5 != None) and (pcent5 !=0) and (pcent5 !=None):
                if abs(cent5 - pcent5) < 20:
                    track5 = cent5 - pcent5
                    state5 = True
                else:
                    cent5 = pcent5
            else:
                state5 = False
                
            pcent1, pcent2, pcent3, pcent4, pcent5 = [cent1, cent2, cent3, cent4, cent5]

# main #5 anchor, others are reference ===========================
            if state5:
                track = track + track5
                missed = track5
                #adjust anchor =====================================
##                if (start != 0) and (start != (track - track5)) and ((track - track5) < 50):
##                    track = cent5 - (start - int(wi/2))
            elif state4:
                track = track + track4
                missed = track4
            elif state3:
                track = track + track3
                missed = track3
            elif state2:
                track = track + track2
                missed = track2
            elif state1:
                track = track + track1
                missed = track1


#            f.write(str(count)+','+str(cent1)+','+str(cent2)+','+str(cent3)+','+str(cent4) +','+ str(cent5)+','+ str(track) +'\n')
            ff.write(str(count)+',' + str(track) +'\n')
            fc.write(str(count)+',' + str(trackc) + '\n')
            cv2.imshow('block5', block5)
            cv2.imshow('block4', block4)
            cv2.imshow('block3', block3)
            cv2.imshow('block2', block2)
            cv2.imshow('block1', block1)

            cv2.imshow('blockc2', blockc2)
            cv2.imshow('blockc1', blockc1)
            cv2.imshow('blockc3', blockc3)
            
            frame_num = str(count)
            gap = str(cent5)
            track_num = str(track)
            cv2.putText(frame, "Time(100ms): "+frame_num, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, gap, (0 ,hi-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, track_num, (0, hi-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            gap2 = str(centc2) # 카메라 중앙에서 벗어난 값
            track_num2 = str(trackc) #변위 추적 값
            if abcent == None:
                abcent = int(wi2/2)
                color = (0,0,255)
                abtext = "Missed"
            else:
                color = (0,255,0)
                abtext = str(abcent)
            cv2.putText(frame2, "Time(100ms): "+frame_num, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame2, gap2, (0 ,hi2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame2, track_num2, (0, hi2-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame2, abtext, (abcent+20, hi2-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            if nozzle_state == 0:
                textc = "Nozzle OFF"
            else:
                textc = "Nozzle ON"
            cv2.putText(frame2, textc, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
            if remain > 0:
                disp = 10
                textcc = "Nozzle state change: "+str(remain) + " (x10ms)"
            if disp > 0:
                cv2.putText(frame2, textcc, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 3, cv2.LINE_AA)
                disp = disp - 1

# 차선의 첫 시작 위치 지정(전방) ==============
            if state5:
                if start == 0:
                    start = cent5 + int(wi/2)
            line = cv2.line(frame, (start, 0), (start, hi), (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(start-int(wi/2)), (wi-50,hi-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)
            
# 차선의 첫 시작 위치 지정(중앙) ==============
            if (startc == 0) and (centc2 !=None):
                startc = centc2 + int(wi2/2)
            line = cv2.line(frame2, (startc, 0), (startc, hi2), (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame2, str(startc-int(wi2/2)), (wi2-50,hi2-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)
            
            cv2.imshow('test', frame)
            cv2.imshow('test2', frame2)
            count = count + 1
            k = cv2.waitKey(10)
            if k==27:
                break
            elif k == 32:
                cv2.waitKey()

        else:
            print('video end')
            ff.close()
            cap_f.release()
            cv2.destroyAllWindows()
            break
            
else:
    print('cannot open file')

ff.close()
fc.close()
cap_f.release()
cv2.destroyAllWindows()
