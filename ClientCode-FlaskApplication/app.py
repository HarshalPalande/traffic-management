from flask import Flask, render_template,request, session, Response,redirect, abort,url_for
import pandas as pd
import os
import cv2,socket
import time
import base64 
from werkzeug.utils import secure_filename
import numpy as np
from threading import Thread
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy 

# Defining upload folder path
UPLOAD_FOLDER = './static/upload'

# Definining allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__ , template_folder='./static/templates')
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
#app = Flask(__name__, template_folder='./static/templates', static_folder='./static/staticFiles')
# define path for upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session 
#app.secret_key = 'You Will Never Guess'
db = SQLAlchemy(app)

class Camera(db.Model):
    id= db.Column(db.Integer,primary_key=True,nullable=True)
    location = db.Column(db.String(150))
    road_length = db.Column(db.Float, nullable=False)
    traffics=db.relationship('Traffic', backref='camera', lazy=True)

    def __repr__(self):
        return f"Camera('{self.id}', '{self.location}', '{self.road_length}' )"

class Traffic(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    AvrgCar = db.Column(db.Integer, nullable=False)
    LOS = db.Column(db.String(1), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)

    def __repr__(self):
        return f"Traffic('{self.AvrgCar}', '{self.LOS}', '{self.date_posted}' )"    

#Generating frames for live streaming 
def generate_frames(id,l):
    length =l
    net = cv2.dnn.readNet('yolov3-320.weights' , 'yolov3-320.cfg')

    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()
        
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    BUFF_SIZE = 65536 
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
    #ip=input("Insert server IP")
    #host_ip=ip

    host_ip='192.168.0.109' 
    
    print(host_ip)
    port = 9999
    socket_address = (host_ip,port)

    message = 'camera:'+str(id)

    while True:#keep sending requests and getting responses
        client_socket.sendto(f"{message}".encode(), socket_address )

        print('Hello message sent to server, requesting camera'+str(id))

        Frame_count =0
        CountVehicules=0
        frameCounter = 0

        packet,_ = client_socket.recvfrom(BUFF_SIZE)
        
        data = base64.b64decode(packet,' /')
        npdata = np.fromstring(data,dtype=np.uint8)
        img = cv2.imdecode(npdata,1)
            
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
        #end if skip condition
        
        #density range calculation (DR)
        DR = int(CountVehicules/(length*Frame_count*3))
        if(DR<=12):
            LOS = "A"
        if(DR>12 and DR<=20):
            LOS = "B"
        if(DR>20 and DR<=30):
            LOS = "C"
        if(DR>30 and DR<=42):
            LOS = "D"
        if(DR>42 and DR<=67):
            LOS = "E"
        if(DR>67):
            LOS = "F"
            
        print(DR,LOS)
        CountVehicules = 0
        Frame_count =0
        
        cv2.rectangle(img,(0,0),(100,40),(255,255,255),-1)
        img = cv2.putText(img,'LOS: '+str(LOS),(3,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        ret,buffer=cv2.imencode('.jpg',img)
        final_img=buffer.tobytes()

        yield(b'--img\r\n'
            b'Content-Type: immage/jpeg\r\n\r\n' + final_img + b'\r\n')
    frameCounter=frameCounter+1    
    socket.close()     

# YOLO object detection function
def detect_object():
    net = cv2.dnn.readNet('yolov3-320.weights' , 'yolov3-320.cfg')
    classes = []
    with open('coco.names' , 'r') as f:
        classes = f.read().splitlines()

    img = cv2.imread('./static/upload/input_image.jpg')
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    colors = np.random.uniform(0, 255, size=(100, 3))

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
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
    font =cv2.FONT_HERSHEY_PLAIN
    count=0 

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            if label in ['bicycle','car','motorbike','train','bus','truck']:
                count=count+1
                color = colors[i%100]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 1.5, (0,0,0), 2)
    
    cv2.rectangle(img,(0,0),(210,40),(255,255,255),-1)
    cv2.putText(img, "vehicles ="+str(count), (3,30), font,2,(0,10,200),2)           
    
    # Write output image (object detection output)
    #output_image_path = os.path.join(app.config['UPLOAD_FOLDER'],'img.filename', 'output_image.jpg')
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    cv2.imwrite(output_image_path, img)
    #return(output_image_path)
    return(count)
    

@app.route('/')
def f1():
    return render_template('page1.html')

@app.route('/options')
def f2():
    return render_template('page2.html')

@app.route('/model_deployment')
def index():
    return render_template('index_upload_and_display_image.html')
 
#@app.route('/model_deployment',  methods=["POST", "GET"])
@app.route('/model_deployment',  methods=["POST"])
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.jpg'))
        session['uploaded_img_file_path']=os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.jpg')
        return render_template('index_upload_and_display_image_page2.html')

    #if request.method == 'GET':
        #return redirect(url_for('show_image'))
 
@app.route('/show_image')
def displayImage():
    #img_file_path = session.get('uploaded_img_file_path', None)
    #return render_template('show_image.html', user_image = img_file_path)
    return render_template('show_image.html')
 
@app.route('/detect_object')
def detectObject():
    # Retrieving uploaded file path from session
    #uploaded_image_path = session.get('uploaded_img_file_path', None)
    # Display image in Flask application web page
    #output_image_path = detect_object('./static/images/input_image.jpg')
    #print(output_image_path)
    #output_image_path=detect_object()
    num=detect_object()
    #return render_template('show_detection.html', user_image = output_image_path)
    return render_template('show_detection.html',num_cars=num)
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
#@app.after_request()
#def add_header(response):
#    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#    response.headers["Pragma"] = "no-cache"
#    response.headers["Expires"] = "0"
#    response.headers['Cache-Control'] = 'public, max-age=0'
#    return response

@app.route('/pageC')
def pageC():
    return render_template('pageC.html',data=Camera.query.filter(Camera.id>0).all())

@app.route('/selected_camera_fromC', methods = ["POST"])
def selected_camera_fromC():
    if request.method == 'POST':
        #get the id of the camera
        global live_camera_id
        global length_road
        live_camera_id_string = request.form.get('text1')
        live_camera_id=int(live_camera_id_string)
        cam_live = Camera.query.filter(Camera.id== live_camera_id).first()
        length_road=cam_live.road_length
        location_live=cam_live.location
        return render_template('pageC2.html',id=live_camera_id,length=length_road,location=location_live)

@app.route('/video')
def video():
    return Response(generate_frames(live_camera_id,length_road),mimetype='multipart/x-mixed-replace; boundary=img')
    

#@app.route('/stop_video')
#def stopvideo():
    #flag_stop=1 
    #return render_template('pageC.html',data=Camera.query.filter(Camera.id>0).all()) 

@app.route('/gotoB')
def gotoB():
    return render_template('pageB.html',data=Camera.query.filter(Camera.id>0).all())

@app.route('/selected_camera_fromB', methods = ["POST"])
def selected_camera_fromB():
    if request.method == 'POST':
        global chosen_camera_id
        chosen_camera_id_string = request.form.get('text1')
        chosen_camera_id=int(chosen_camera_id_string)
        all_objects_from_chosen_camera = Traffic.query.filter(Traffic.camera_id== chosen_camera_id).all()
        return render_template("pageB2.html",cameraid=chosen_camera_id, data=all_objects_from_chosen_camera)

@app.route('/selected_time1_fromB2', methods = ["POST"])
def selected_time1_fromB2():
    if request.method == 'POST':
        global chosen_time1
        chosen_time1_string=request.form.get('text2')
        chosen_time1 = datetime.strptime(chosen_time1_string, '%Y-%m-%d %H:%M:%S.%f') 

        all_objects_from_chosen_camera = Traffic.query.filter(Traffic.camera_id== chosen_camera_id).all()
        
        #global all_times_after_chosen_time1 
        all_objects_times_after_chosen_time1= objects_time2(all_objects_from_chosen_camera)
        return render_template("pageB3.html",cameraid=chosen_camera_id, times1=chosen_time1, times2=all_objects_times_after_chosen_time1)

def objects_time2(all_objects_from_chosen_camera):
    liste=[]
    for x in all_objects_from_chosen_camera:
                if (x.date_posted > chosen_time1):
                    liste.append(x)
    return liste

def list_of_los(all_objects_between_t1andt2):
    liste=[]
    A=B=C=D=E=F=0
    for w in all_objects_between_t1andt2:
        if(w.LOS=='A'):
            A=A+1
        if(w.LOS=='B'):
            B=B+1
        if(w.LOS=='C'):
            C=C+1
        if(w.LOS=='D'):
            D=D+1
        if(w.LOS=='E'):
            E=E+1
        if(w.LOS=='F'):
            F=F+1
    return A,B,C,D,E,F

def list_of_nbcars(all_objects_between_t1andt2):
    liste=''   
    for x in all_objects_between_t1andt2:
        liste=liste+','+str(x.AvrgCar)
    return liste

def list_of_times(all_objects_between_t1andt2):
    liste=''
    for x in all_objects_between_t1andt2:
        liste=liste+'&'+str(x.date_posted)
    return liste

@app.route('/selected_time2_fromB3', methods = ["POST"])
def selected_time2_fromB2():
    if request.method == 'POST':
        global chosen_time2
        all_objects_from_chosen_camera = Traffic.query.filter(Traffic.camera_id== chosen_camera_id).all()
        all_objects_times_after_chosen_time1 = objects_time2(all_objects_from_chosen_camera)
        chosen_time2_string=request.form.get('text3')
        chosen_time2 = datetime.strptime(chosen_time2_string, '%Y-%m-%d %H:%M:%S.%f')
        global all_objects_between_t1andt2
        all_objects_between_t1andt2 = objects_dashboard(all_objects_times_after_chosen_time1)
   
        #occurence of all values of LOS
        A,B,C,D,E,F=list_of_los(all_objects_between_t1andt2)
        
        #list of number of cars
        list_nbcars=list_of_nbcars(all_objects_between_t1andt2)

        #list of times
        list_times=list_of_times(all_objects_between_t1andt2)
        
        return render_template('pageB4.html',cameraid=chosen_camera_id, times1=chosen_time1, times2=chosen_time2 ,A=A,B=B,C=C,D=D,E=E,F=F,nbCars=list_nbcars, times=list_times)
        

def objects_dashboard(all_objects_times_after_chosen_time1):
    liste=[]
    for x in all_objects_times_after_chosen_time1:
        if(x.date_posted <chosen_time2):
            liste.append(x)
    return liste

@app.route('/gotoD')
def pageD():
    return render_template('pageD.html',data=Camera.query.filter(Camera.id>0).all())

@app.route('/selected_camera_fromD', methods = ['POST','GET'])
def selected_camera_fromD():
    chosen_camera_id_string = request.form.get('text1')
    chosen_camera_id=int(chosen_camera_id_string)
    chosen_period_string = request.form.get('text2')
    chosen_period=int(chosen_period_string)
    jumps=chosen_period*2 #because the databse was updated every 30 seconds or half a minute
    j=0
    A=B=C=D=E=F=0
    strA=''
    strB=''
    strC=''
    strD=''
    strE=''
    strF=''
    strTimesSelected=''
    all_objects_from_chosen_camera = Traffic.query.filter(Traffic.camera_id== chosen_camera_id).all()

    for w in all_objects_from_chosen_camera:
        if j==jumps:  
            strA=strA+','+str(A)
            strB=strB+','+str(B)
            strC=strC+','+str(C)
            strD=strD+','+str(D)
            strE=strE+','+str(E)
            strF=strF+','+str(F)
            strTimesSelected=strTimesSelected+'&'+str(w.date_posted)
            A=B=C=D=E=F=0
            j=1
        else:
            if(w.LOS=='A'):
                A=A+1
            if(w.LOS=='B'):
                B=B+1
            if(w.LOS=='C'):
                C=C+1
            if(w.LOS=='D'):
                D=D+1
            if(w.LOS=='E'):
                E=E+1
            if(w.LOS=='F'):
                F=F+1
            j=j+1
   
    return render_template('pageD2.html',cameraid=chosen_camera_id,analytics=chosen_period,nbA=strA,nbB=strB,nbC=strC,nbD=strD,nbE=strE,nbF=strF,times=strTimesSelected)

@app.route('/LOSinfo')
def LOSinfoD():
    return render_template('LOSinfo.html')

@app.after_request
def add_header(r):
    r.headers["Cache-Control"]="no-cache, no-store, must-revalidate"
    r.headers["Pragma"]="no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control']='public, max_age=0'
    return r

if __name__=='__main__':
    final_img=''
    flag_stop=0
    #real_time_thread = Thread(target=generate_frames)
    #real_time_thread.start()
    #generate_frames()
    app.run(debug = True)
    
