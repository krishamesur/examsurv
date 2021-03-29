from flask import Flask, render_template, Response ,request
from camera import VideoCamera
import cv2
import numpy as np
import pyttsx3
from Detectface import DetectFace
app = Flask(__name__)

def gen1(camera):
    sampleNum = 0
    
    
    
    while True:
        sampleNum=sampleNum+1
        frame = camera.get_frame(sampleNum)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if cv2.waitKey(100) & sampleNum>=60:
            with app.app_context(), app.test_request_context():
                 return redirect(url_for('train'))
            print("way passed")    
    
    print('Training Done')
    camera.__del__()    

def gen2(camera):
    
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath); 
   
 
    while True:
            frame = camera.get_frame(faceCascade)
        
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            
@app.route('/')
def index():
    return render_template('index.html')            

@app.route('/detect_face',methods=['POST'])
def detect_face():
    return Response(gen2(DetectFace()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
