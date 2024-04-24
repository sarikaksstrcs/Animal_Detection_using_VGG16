from tensorflow.keras.models import load_model
import keras.utils as image
import numpy as np

# Load the multi-class classification model
model = load_model('animal_classification_model.h5')

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values

# Function to predict whether the image contains an elephant, tiger, or other animal
def predict_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    class_labels = ['Elephant',  'Other Animal','Tiger']  # Assuming 'Other Animal' is the third class
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

sample_image_path = 'anushka.jpg'

# Make prediction
prediction = predict_image(sample_image_path)
print("Prediction:", prediction)