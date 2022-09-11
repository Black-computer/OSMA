import numpy as np
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import math
import time
from sympy import *

def calculate_point(x0,y0,x1,y1):
    result = math.sqrt(math.pow(x0-x1, 2)+math.pow(y0-y1, 2))
    return result


def calculate_contours_average(contours_now):
    global grey
    global grey_all
    global grey_average
    grey = 0
    grey_all = 0
    grey_average = 0
    for i in range(len(contours_now)):
        grey = fgmask[contours_now[i][0][1], contours_now[i][0][0]]
        grey_all = grey + grey_all
        grey_average = grey_all / float(len(contours_now))
    return grey_average

def calculate_Segmentation_Threshold(contours_bigist,dis):
    global grey
    global grey_all
    global grey_average
    global abandan_point
    global shadow_point
    global lens
    global label
    grey = 0
    grey_all = 0
    grey_average = 0
    for i in range(len(contours_bigist)):
        grey = fgmask[contours_bigist[i][0][1], contours_bigist[i][0][0]]
        grey_all = grey + grey_all
        grey_average = grey_all / float(len(contours_bigist))

    abandan_point = Symbol('x')
    shadow_point = Symbol('y')
    lens = len(contours_bigist)

    dict = solve([255*abandan_point+127*shadow_point-grey_average*lens, abandan_point+shadow_point-lens], [abandan_point,shadow_point])
    label = list(dict.values())
    percent = label[0]/lens
    Segmentation_Threshold = 127+64*percent*2


    weight = (1-dis/100)*(1-percent)
    Segmentation_Threshold_frame = 127+64*(percent-weight)*2
    if Segmentation_Threshold_frame<0:
        Segmentation_Threshold_frame = 0

    return Segmentation_Threshold,Segmentation_Threshold_frame

def calculate_IOU(xmin2,ymin2,xmax2,ymax2,z):
    xml_path = 'D:/Abanden/experiment/3D_printing/xml/xml/{}.xml'.format(str(z))
    tree = ET.parse(xml_path)
    rect = {}
    line = ""
    root = tree.getroot()
    with open('blog.txt', 'w', encoding='utf-8') as f1:

        for name in root.iter('path'):
            rect['path'] = name.text
        for ob in root.iter('object'):

            for bndbox in ob.iter('bndbox'):

                for xmin in bndbox.iter('xmin'):
                    rect['xmin'] = xmin.text
                    xmin1 = int(rect['xmin'])
                for ymin in bndbox.iter('ymin'):
                    rect['ymin'] = ymin.text
                    ymin1 = int(rect['ymin'])
                for xmax in bndbox.iter('xmax'):
                    rect['xmax'] = xmax.text
                    xmax1 = int(rect['xmax'])
                for ymax in bndbox.iter('ymax'):
                    rect['ymax'] = ymax.text
                    ymax1 = int(rect['ymax'])
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])


    centre_point_x1 = xmin1 + (xmax1 - xmin1) / 2
    centre_point_y1 = ymax1 - (ymax1 - ymin1) / 2
    centre_point_x2 = xmin2 + (xmax2 - xmin2) / 2
    centre_point_y2 = ymax2 - (ymax2 - ymin2) / 2

    distance_point = math.sqrt((centre_point_x2 - centre_point_x1)**2 + (centre_point_y2 - centre_point_y1)**2)


    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou, distance_point



np.set_printoptions(threshold=np.inf)

cap = cv2.VideoCapture('D:/Abanden/experiment/3D_printing/Video/3D_printing.mp4')
z = 0
object0 = 0
object1 = 0
distance_0 = 0
distance_1 = 0
distance_average = 0.01
Abandon = 0
distance_var = 35
centre_grey = 0
T = 0.6
T_first = 0.3
target = {}
target_2 = {}
x = 0
FLAG = 0
FLAG_abandon = 0

target_3 = {}
y = 0
grey = 0
grey_all = 0
grey_average = 0
abandan_point = 0
shadow_point = 0
lens = 0
label = []


#对比数据
IOU_average = 0
IOU_percent = 0
distance_percent = 0


start_frame = 663
first_frame = 664
sec_frame = 665
the_frame = 666
end_frame = 667


saveFile = "D:/Abanden/experiment/shuihu/RGB/{}.jpg".format(str(z))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


fgbg = cv2.createBackgroundSubtractorMOG2()
sd = cv2.createShapeContextDistanceExtractor()

while (True):
    ret, frame = cap.read()
    if ret == False:
        break

    fgmask = fgbg.apply(frame)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if z > start_frame:
        print(time.time())  # 返回当前的时间戳（1970以后）

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for index0, i in enumerate(contours):
            area = cv2.contourArea(i)

            if(area<10):
                 del contours[index0:]
                 break


        if (z>start_frame)&(z<end_frame):

            for index, i in enumerate(contours):
                flag = index + 1
                for index2, j in enumerate(contours):
                    if index2 == flag:

                        d_del = cv2.matchShapes(contours[index],contours[index2],cv2.CONTOURS_MATCH_I3,0)

                        if d_del < T:
                            cnt_true = contours[index]
                            contours_grey0 = fgmask[cnt_true[1][0][1], cnt_true[1][0][0]]
                            cnt_False = contours[index2]
                            contours_grey1 = fgmask[cnt_False[1][0][1], cnt_False[1][0][0]]


                            contours_grey_average_target = calculate_contours_average(contours[index])
                            contours_grey_average_shadow = calculate_contours_average(contours[index2])

                            Segmentation_Threshold, Segmentation_Threshold_frame = calculate_Segmentation_Threshold(
                                contours[0], 0)


                            if ((contours_grey_average_target >= Segmentation_Threshold) & (contours_grey_average_shadow < Segmentation_Threshold)):

                                if (len(target) != 0)&(z == sec_frame):
                                    rect_now = cv2.minAreaRect(contours[index])
                                    centre_list = list(rect_now[0])
                                    centre_int = list(map(int, centre_list))
                                    rect_now2 = cv2.minAreaRect(contours[index2])
                                    centre_list2 = list(rect_now2[0])
                                    centre_int2 = list(map(int, centre_list2))
                                    x0 = centre_int[0]
                                    y0 = centre_int[1]
                                    x1 = centre_int2[0]
                                    y1 = centre_int2[1]
                                    for i in range(0,(len(target)//6)):
                                        x2 = target[6*i+1]
                                        y2 = target[6*i+2]
                                        x3 = target[6*i+4]
                                        y3 = target[6*i+5]
                                        distance_0 = calculate_point(x0, y0, x2, y2)
                                        distance_1 = calculate_point(x1, y1, x3, y3)

                                        if (distance_0 < distance_var) & (distance_1 < distance_var):
                                            target_2[y] = index
                                            y = y + 1
                                            target_2[y] = centre_int[0]
                                            y = y + 1
                                            target_2[y] = centre_int[1]
                                            y = y + 1
                                            target_2[y] = index2
                                            y = y + 1
                                            target_2[y] = centre_int2[0]
                                            y = y + 1
                                            target_2[y] = centre_int2[1]
                                            y = y + 1

                                        else:
                                            print("wu")
                                        # continue
                                elif(z == first_frame):

                                    rect_now = cv2.minAreaRect(contours[index])
                                    centre_list = list(rect_now[0])
                                    centre_int = list(map(int, centre_list))
                                    rect_now2 = cv2.minAreaRect(contours[index2])
                                    centre_list2 = list(rect_now2[0])
                                    centre_int2 = list(map(int, centre_list2))
                                    target[x] = index
                                    x = x + 1
                                    target[x] = centre_int[0]
                                    x = x + 1
                                    target[x] = centre_int[1]
                                    x = x + 1
                                    target[x] = index2
                                    x = x + 1
                                    target[x] = centre_int2[0]
                                    x = x + 1
                                    target[x] = centre_int2[1]
                                    x = x + 1

                        flag = flag + 1
                        continue
                    else:
                        continue
            if (((len(target_2) / 6) == 1) & (z > first_frame)):

                Abandon = target_2[0]
                Abandon_contours = contours[Abandon]
                rect_init = cv2.minAreaRect(contours[Abandon])
                centre_Init = rect_init[0]

                x, y, w, h = cv2.boundingRect(contours[Abandon])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                #IOU计算
                IOU, Two_point_distance = calculate_IOU(x, y, (x + w), (y + h), z)
                if IOU > 0.5:
                    IOU_percent = IOU_percent + 1


                IOU_average = IOU_average + IOU
                print("IOU_percent=", IOU_percent)
                print("IOU_average=", IOU_average)
                print("IOU=---------------------------------------------------------------", IOU)


                if Two_point_distance < 20:
                    distance_percent = distance_percent + 1
                print("distance_percent=", distance_percent)


            elif(((len(target_2) / 6) > 1)&(z > sec_frame)):
                print("不止一对适配点")
                y = 0
                for index, i in enumerate(contours):
                    flag = index + 1
                    for index2, j in enumerate(contours):
                        if index2 == flag:
                            flag = flag + 1

                            d_del = cv2.matchShapes(contours[index], contours[index2], cv2.CONTOURS_MATCH_I3, 0)

                            if d_del < (T / 2):
                                cnt_true = contours[index]
                                contours_grey0 = fgmask[cnt_true[0][0][1], cnt_true[0][0][0]]
                                cnt_False = contours[index2]
                                contours_grey1 = fgmask[cnt_False[0][0][1], cnt_False[0][0][0]]

                                contours_grey_average_target = calculate_contours_average(contours[index])
                                contours_grey_average_shadow = calculate_contours_average(contours[index2])

                                Segmentation_Threshold, Segmentation_Threshold_frame = calculate_Segmentation_Threshold(contours[0],0)

                                if ((contours_grey_average_target >= Segmentation_Threshold) & (contours_grey_average_shadow < Segmentation_Threshold)):

                                    if (len(target_2) != 0) & (z == the_frame):

                                        rect_now = cv2.minAreaRect(contours[index])
                                        centre_list = list(rect_now[0])
                                        centre_int = list(map(int, centre_list))
                                        rect_now2 = cv2.minAreaRect(contours[index2])
                                        centre_list2 = list(rect_now2[0])
                                        centre_int2 = list(map(int, centre_list2))
                                        x0 = centre_int[0]
                                        y0 = centre_int[1]
                                        x1 = centre_int2[0]
                                        y1 = centre_int2[1]
                                        for i in range(0, (len(target_2) // 6 - 1)):
                                            x2 = target_2[6 * i + 1]
                                            y2 = target_2[6 * i + 2]
                                            x3 = target_2[6 * i + 4]
                                            y3 = target_2[6 * i + 5]
                                            distance_0 = calculate_point(x0, y0, x2, y2)
                                            distance_1 = calculate_point(x1, y1, x3, y3)

                                            if (distance_0 < distance_var) & (distance_1 < distance_var):
                                                target_3[y] = index
                                                y = y + 1
                                                target_3[y] = centre_int[0]
                                                y = y + 1
                                                target_3[y] = centre_int[1]
                                                y = y + 1
                                                target_3[y] = index2
                                                y = y + 1
                                                target_3[y] = centre_int2[0]
                                                y = y + 1
                                                target_3[y] = centre_int2[1]
                                                y = y + 1

                if ((len(target_3) / 6) == 1):

                    Abandon = target_3[0]
                    Abandon_contours = contours[Abandon]
                    rect_init = cv2.minAreaRect(contours[Abandon])
                    centre_Init = rect_init[0]

                    x, y, w, h = cv2.boundingRect(contours[Abandon])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # IOU计算
                    IOU, Two_point_distance = calculate_IOU(x, y, (x + w), (y + h), z)
                    if IOU > 1:
                        IOU_percent = IOU_percent + 1


                    IOU_average = IOU_average + IOU
                    print("IOU_percent=", IOU_percent/200)
                    print("IOU_average=", IOU_average)
                    print("IOU=---------------------------------------------------------------", IOU)

                    # 点间距计算
                    if Two_point_distance < 50:
                        distance_percent = distance_percent + 1
                    print("distance_percent=", distance_percent/200)
                else:
                    print("疑似抛洒物定位失败")



        if (Abandon != 0) & (z>the_frame):
            if(FLAG_abandon != 0):
                if (FLAG_abandon >= 10) & ((distance_average / FLAG_abandon) < 2.0):

                    x, y, w, h = cv2.boundingRect(rest_contours)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # IOU计算
                    if z < 865:
                        IOU, Two_point_distance = calculate_IOU(x, y, (x + w), (y + h), z)
                        if IOU > 1:
                            IOU_percent = IOU_percent + 1

                        IOU_average = IOU_average + IOU
                        print("IOU_percent=", IOU_percent/200)
                        print("IOU_average=", IOU_average)
                        print("IOU=---------------------------------------------------------------", IOU)

                        # 点间距计算
                        if Two_point_distance < 50:
                            distance_percent = distance_percent + 1
                        print("distance_percent=", distance_percent/200)


                    z = z + 1
                    print("z=",z)
                    continue

            for index0, i in enumerate(contours):
                area = cv2.contourArea(i)

                if (area < 10):
                    del contours[index0:]
                    break
            for index3, i in enumerate(contours):
                rect_now = cv2.minAreaRect(contours[index3])
                centre_list = list(rect_now[0])
                centre_int = list(map(int, centre_list))

                if FLAG == 1:
                    centre_Init_list = centre_Init_list_now
                    centre_Init_int = centre_Init_int_now
                    Abandon_contours = Abandon_contours_now

                else:
                    centre_Init_list = list(centre_Init)
                    centre_Init_int = list(map(int, centre_Init_list))

                x0 = centre_int[0]
                y0 = centre_int[1]
                x1 = centre_Init_int[0]
                y1 = centre_Init_int[1]
                distance_0 = calculate_point(x0, y0, x1, y1)

                d_del = cv2.matchShapes(Abandon_contours, contours[index3], cv2.CONTOURS_MATCH_I3, 0)

                centre_grey = fgmask[centre_int[1],  centre_int[0]]

                contours_grey_average_target = calculate_contours_average(contours[index3])
                Segmentation_Threshold, Segmentation_Threshold_frame = calculate_Segmentation_Threshold(contours[0], distance_0)

                if (distance_0 < 40)&(contours_grey_average_target>=Segmentation_Threshold_frame):
                    if distance_0 < 1:
                        FLAG_abandon = FLAG_abandon + 1
                        distance_average = distance_0 + distance_average
                        if (FLAG_abandon >= 10)& ((distance_average//FLAG_abandon) < 2.0):
                            rest_abandon = cv2.minAreaRect(contours[index3])
                            rest_contours = contours[index3]
                    FLAG = 1
                    rect = cv2.minAreaRect(contours[index3])

                    x, y, w, h = cv2.boundingRect(contours[index3])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # IOU计算
                    if z < 865:
                        IOU, Two_point_distance = calculate_IOU(x, y, (x + w), (y + h), z)
                        if IOU > 1:
                            IOU_percent = IOU_percent + 1

                        IOU_average = IOU_average + IOU
                        print("IOU_percent=", IOU_percent/200)
                        print("IOU_average=", IOU_average)
                        print("IOU=---------------------------------------------------------------", IOU)

                        # 点间距计算
                        if Two_point_distance < 50:
                            distance_percent = distance_percent + 1
                        print("distance_percent=", distance_percent/200)


                    centre_Init_list_now = list(rect[0])
                    centre_Init_int_now = list(map(int, centre_Init_list_now))

                    Abandon_contours_now = contours[index3]

                    break
    z = z + 1
    print("z=",z)
    # cv2.imshow('frame', frame)
    # cv2.imshow('fgmask', fgmask)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
