
import cv2
import numpy as np

from predict import predict_image
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Perform prediction on the frame
    prediction = predict_image(frame)
    
    # Display the prediction
    cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with prediction
    cv2.imshow('Frame with Prediction', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()