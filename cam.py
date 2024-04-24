import smtplib
from email.mime.text import MIMEText
from time import strftime

import cv2
import datetime
import numpy as np
# from SJV.DBConnection import *
import pymysql
from django.db import connection
from predict import predict_image

import os
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# classes = ["Sign Board"]
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net1.getLayerNames()
ln = [ln[i[0] - 1] for i in net1.getUnconnectedOutLayers()]


live_Camera = cv2.VideoCapture(0)

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

def main_code():
    pr=""
    ret, first_frame = live_Camera.read()
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Compute the total number of pixels in the frame
    total_pixels = first_frame_gray.shape[0] * first_frame_gray.shape[1]
    prcount=0
    (W, H) = (None, None)
    while (live_Camera.isOpened()):

        ret, frame = live_Camera.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current frame and the first frame
        frame_diff = cv2.absdiff(frame_gray, first_frame_gray)

        # Compute the mean absolute difference (MAD) between corresponding pixels
        diff_value = np.sum(frame_diff) / total_pixels
        if diff_value > 35:
            print(diff_value,"====================>")
            print(diff_value,"====================>")
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net1.setInput(blob)
            #
            layerOutputs = net1.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            pl = []
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.35:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([1,2,3,4])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4,
                                    0.5)
            pcount = 0
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]

                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                               confidences[i])
                    if LABELS[classIDs[i]] == "person" or classIDs[i]==14 :
                        # or classIDs[i]==15 or classIDs[i]==16 :
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        #  15 and 16 c&d
                        pl.append("1")
                        pcount += 1
                        print(LABELS[classIDs[i]])
                        # cropped_image = frame[y:y + h, x:x + w]

                        # res = predict("sample.png")
                        # print(res,"===========")

                        # cv2.putText(frame, "person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, print_color, 2)
                    if classIDs[i]==15 or classIDs[i]==16:

                        if confidences[i]>0.55:
                            pl.append("1")
                            pcount += 1
                            print(LABELS[classIDs[i]])
            if len(pl)==0:
                # cv2.putText(frame, "Fire Detected !", (300, 60), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
                # fn = 'sample.jpg'
                fn = strftime("%Y%m%d%H%M%S") + ".png"
                # #fn1=r"D:\final_project\ForestFire\ForestFire\media/" + fn

                fn1=r"/" + fn
                # # cv2.imwrite(r"D:\final_project\ForestFire\ForestFire\media/" + fn, frame)
                # cv2.imwrite(fn,frame)
                cv2.imwrite(r"/" + fn, frame)
                res1=predict_image(fn1)

                if res1!="zzzzz" and res1!="zzzzz1":
                   print("Animal detected")
                   prcount+=1
                   if  pr == res1 or prcount>1:
                    # Using cv2.putText() method
                    frame = cv2.putText(frame, 'Animal Detected', org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                    import winsound
                    frequency = 2500  # Set Frequency To 2500 Hertz
                    duration = 2000  # Set Duration To 1000 ms == 1 second
                    winsound.Beep(frequency, duration)

                    send_noti()

                    print("******************************************************************************************")
                    print("******************************************************************************************")
                    print("******************************************************************************************")
                    print("******************************************************************************************")
                    print("******************************************************************************************")
                    print("******************************************************************************************")

                    v=4

                    con=pymysql.connect(host='localhost',port=3308,user='root',password='12345678',db='forestfiredetection')
                    cmd=con.cursor()
                    cmd.execute("SELECT MINUTE (TIMEDIFF(CURTIME(), `time`)) AS minu FROM `forest_app_notification_table` WHERE CAMERA_id=1 ORDER BY id DESC LIMIT 1")
                    cmd.fetchone()

                    cmd.execute("SELECT MINUTE (TIMEDIFF(CURTIME(), `time`)) AS minu FROM `forest_app_notification_table` WHERE CAMERA_id='"+str(v)+"' ORDER BY id DESC LIMIT 1")
                    res=cmd.fetchone()
                    print(res,"000000000000000000000000")
                    print("detected")
                    if res is None:

                        cmd.execute("INSERT INTO `forest_app_notification_table` VALUES(NULL,CURDATE(),curtime(),'"+fn+"','animal',4)")
                        con.commit()
                    else:
                        mi=res[0]
                        print("minute",mi)
                        if mi>=1:

                            cmd.execute("select * from forest_app_notification_table where CAMERA_id=4 and date=curdate() and time=curtime()")


                            res=cmd.fetchone()
                            if res is None:
                                cmd.execute(
                                    "INSERT INTO `forest_app_notification_table` VALUES(NULL,CURDATE(),curtime(),'" + fn + "','animal',4)")
                                con.commit()
                else:
                    print(res1,"=================")
                    prcount=0
                pr=res1
            else:
                pr="na"
                prcount=0
        cv2.imshow("Animal Detection", frame)

        if cv2.waitKey(10) == 27:
            print("break")
            break
    live_Camera.release()

    cv2.destroyAllWindows()




def send_noti():

    try:
        print("1","gggggggggggggggggggggg")

        qry = "SELECT `email` FROM `forest_app_officer_table`"
        con = pymysql.connect(host='localhost', port=3308, user='root', password='12345678', db='forestfiredetection')
        cmd = con.cursor()
        cmd.execute(qry)
        s=cmd.fetchall()
        print(s, "=============")
        for i in s:
            eml=i[0]
            try:
                gmail = smtplib.SMTP('smtp.gmail.com', 587)
                gmail.ehlo()
                gmail.starttls()
                gmail.login('anushkacet@gmail.com', 'vyyrwxtdxwugvtqg')
                print("login=======")
            except Exception as e:
                print("Couldn't setup email!!" + str(e))
            msg = MIMEText("Animal Detect on camera no 4")
            print(msg)
            msg['Subject'] = 'Animal Detected'
            msg['To'] = i[0]
            msg['From'] = 'anushkacet@gmail.com'

            print("ok====")

            try:
                gmail.send_message(msg)
            except Exception as e:
                print(e)
                pass

    except Exception as e:
        print(e)

main_code()