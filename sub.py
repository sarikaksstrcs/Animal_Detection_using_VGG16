import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from time import strftime
import pymysql


import winsound
import time

from tensorflow.keras.models import load_model

# Load your animal classification model
model = load_model('animal_classification_model.h5')

live_Camera = cv2.VideoCapture(0)

# Font settings for annotation
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (0, 0, 255)
thickness = 2

# Function to preprocess image for prediction
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    return img_array / 255.0  # Normalize pixel values

def predict_animal(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    class_labels = ['Elephant',  'Other Animal','Tiger']  # Assuming 'Other Animal' is the third class
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

def send_notification(animal_prediction):
    try:
        # qry = "SELECT `email` FROM `forest_app_officer_table`"
        # con = pymysql.connect(host='localhost', port=3308, user='root', password='12345678', db='forestfiredetection')
        # cmd = con.cursor()
        # cmd.execute(qry)
        # recipients = cmd.fetchall()
        
        for recipient in [["recipients"]]:
            recipient_email = recipient[0]
            try:
                gmail = smtplib.SMTP('smtp.gmail.com', 587)
                gmail.ehlo()
                gmail.starttls()
                gmail.login('anushkacet@gmail.com', 'vyyrwxtdxwugvtqg')
            except Exception as e:
                print("Couldn't setup email!!" + str(e))
                print("Animal Detected on camera no 4"+animal_prediction)
            msg = MIMEText("Animal Detected on camera no 4"+animal_prediction)
            msg['Subject'] = 'Animal Detected'
            msg['To'] = 'sarikaksstrcs@gmail.com'
            msg['From'] = 'anushkacet@gmail.com'

            try:
                gmail.send_message(msg)
            except Exception as e:
                print(e)
                pass

    except Exception as e:
        print(e)

def main_code():
    while live_Camera.isOpened():
        ret, frame = live_Camera.read()
        if not ret:
            break

        # Object detection code...
        
        # Animal detection
        animal_prediction = predict_animal(frame)
        
        print(animal_prediction)
        if animal_prediction != 'Other Animal':
          

            print("Alert!!! Animal Detected........ ",animal_prediction)

            
            
            cv2.putText(frame, animal_prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
            
            freq = 2500
            dur = 2000

            send_notification(animal_prediction)
            winsound.Beep(freq,dur)
            time.sleep(10)
            

        cv2.imshow("Animal Detection", frame)
        if cv2.waitKey(10) == 27:
            break

    live_Camera.release()
    cv2.destroyAllWindows()

main_code()
