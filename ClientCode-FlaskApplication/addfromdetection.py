import cv2
import numpy as np
import time

from datetime import datetime
from app import db
#db.create_all()
from app import Traffic
id_=0
update_per = 30
length =0.19611742 #length of the road covered by the camera

net = cv2.dnn.readNet('yolov3-320.weights' , 'yolov3-320.cfg')

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()
    
#cap =cv2.VideoCapture('Road traffic video for object recognition1.mp4')
#cap =cv2.VideoCapture('Road traffic video for object recognition2.mp4')
cap =cv2.VideoCapture('34 mins one road 480p x0.5.mp4')

#cap = cap =cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

start_time = time.time() 

Frame_count =0
CountVehicules=0

while True:
    #dT = time.time() - tlast
    #print(dT) 
    #fps = 1/dT 
    #tlast = time.time() 
    
    #period=0 or take time
    #carcount=0
    #framecount=0
    #while(period<update_per)
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    #pass blob to model
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    
    #Execute forward pass
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    Count=0
    
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            if label in ['bicycle','car','motorbike','train','bus','truck']:
                Count=Count+1
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
                CountVehicules=CountVehicules+1
    #carcount = carcount + count
    Frame_count = Frame_count +1
    
    #cv2.putText(img, "vehicles = "+str(count), (5,20), font,2,(0,0,255),2) 
    #cv2.putText(img, "fps = "+str(int(fps)), (5,50), font,2,(0,0,255),2) 
            
    cv2.imshow('Video', img)
    
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# the 'q' button is set as the quitting button you may use any desired button of your choice

#press ESC to exit
#if cv2.wait(25) & 0xFF == 27:
#break
    if (time.time() - start_time)>update_per:
        #density range calculation (DR)
        DR = int(CountVehicules/(length*Frame_count*3))
        if (DR<=12):
            los = "A"
        if (DR>12 and DR<=20) :
            los = "B"
        if (DR>20 and DR<=30) :
            los = "C"
        if (DR>30 and DR<=42) :
            los = "D"
        if (DR>42 and DR<=67) :
            los = "E"
        if (DR>67):
            los = "F"
        
        #traffic_to_add=Traffic(AvrgCar=CountVehicules/(Frame_count*3), LOS=los, date_posted=datetime.now(), camera_id=1)
        #traffic_to_add=Traffic(AvrgCar=CountVehicules/(Frame_count*3), LOS=los, date_posted=datetime.now(), camera_id=2)
        traffic_to_add=Traffic(AvrgCar=CountVehicules/(Frame_count*3), LOS=los, date_posted=datetime.now(), camera_id=3)
        db.session.add(traffic_to_add)
        #db.session.commit()
        print(DR,los)
        CountVehicules = 0
        Frame_count =0
        start_time=time.time()
    
    db.session.commit()
    if cv2.waitKey(1)& 0xFF == ord('0'):
        break

cap.release()
#db.session.commit()
cv2.destroyAllWindows()