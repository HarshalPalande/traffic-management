import socket
import threading
import cv2
import imutils
import base64 #to convert the image data to text format

def capture1():
    vid1 = cv2.VideoCapture("Road traffic video for object recognition1.mp4")
    global frame1
    while(vid1.isOpened()):
        _1,frame1 = vid1.read()
    
def capture2():
    vid2 = cv2.VideoCapture("Road traffic video for object recognition2.mp4")
    global frame2
    while(vid2.isOpened()):
        _2,frame2 = vid2.read()

def capture3():
    vid3 = cv2.VideoCapture("34 mins one road 480p x0.75.mp4")
    global frame3
    while(vid3.isOpened()):
        _3,frame3 = vid3.read()

def handle_request(data,client_address):
    if data.decode().startswith("camera:"):
        camera=data.decode()[data.decode().index(":")+1]
    if camera=='1':
        resp = imutils.resize(frame1,width=400)
    if camera=='2':
        resp = imutils.resize(frame2,width=400)
    if camera=='3':
        resp = imutils.resize(frame3,width=400)
    
    with server_socket_lock:
        encoded,buffer = cv2.imencode('.jpg',resp,[cv2.IMWRITE_JPEG_QUALITY,80])
        message = base64.b64encode(buffer)
        server_socket.sendto(message,client_address)
        

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

server_socket_lock=threading.Lock()

server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)

host_ip="0.0.0.0"
print(host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)

def main():
    print('Listening at:',socket_address)
    try:
        while True:
            try:
                data,client_address=server_socket.recvfrom(BUFF_SIZE)
                c_thread=threading.Thread(target=handle_request, args=(data,client_address))
                c_thread.daemon=True
                c_thread.start()

            except OSError as err:
                print("ERROR")

    #except OSError as err2:
    except KeyboardInterrupt:
        print("Shutting down server...")
        server_socket.close()


Thread1=threading.Thread(target=capture1)
Thread2=threading.Thread(target=capture2)
Thread3=threading.Thread(target=capture3)
Thread4=threading.Thread(target=main)

Thread1.start()
Thread2.start()
Thread3.start()
Thread4.start()

Thread1.join()
Thread2.join()
Thread3.join()
Thread4.join()