import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask,request,jsonify











emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
emotionSeries=[]


# # start the webcam feed
# # cap = cv2.VideoCapture(0)
# # add="http://192.168.1.4:8080/video"
# # cap.open(add)

# # # pass here your video path
# # # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/







def video_prediction_emotion(f):
     
     cap = cv2.VideoCapture("upload\\v1.mp4")
     cap.set(cv2.CAP_PROP_POS_FRAMES, f)

   

     for i in range (1):
      
        
    # Find haar cascade to draw bounding box around face
       ret, frame = cap.read()
       frame = cv2.resize(frame, (1280, 720))
       if not ret:
        break
       face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
       gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
       num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
       for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
     
     print(f)
      
     return emotion_dict[maxindex]



app = Flask(__name__)

@app.route('/emotion/<int:frame>')
def emotion(frame):
    return video_prediction_emotion(f=frame)

if __name__ == '__main__':
    app.run( debug=True)































# import cv2
# import eventlet
# from flask import Flask
# from flask_socketio import SocketIO, emit

# app = Flask(__name__)
# socketio = SocketIO(app, async_mode='eventlet')

# eventlet.monkey_patch()

# def capture_frames():
#     camera = cv2.VideoCapture('upload//v1.mp4')
#     while True:
#         ret, frame = camera.read()
#         if ret:
#             # Process the frame here
#             # ...
#             encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
#             eventlet.sleep(0)
#             socketio.emit('frame', encoded_frame)

# @socketio.on('connect')
# def on_connect():
#     socketio.start_background_task(capture_frames)

# if __name__ == '__main__':
#     socketio.run(app,port=5000,debug=True)
