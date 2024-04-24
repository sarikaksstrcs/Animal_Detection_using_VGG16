# %%
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model



# %%
input_shape = (224, 224)
batch_size = 32
num_classes = 3           # Three classes: elephant, tiger, others


# %%
# Unzip the files
def unzip_file(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


elephant_zip = 'elephant.zip'
tiger_zip = 'tiger.zip'
others_zip = 'others.zip'

data_dir = 'animals'

# Make sure the extract directory exists
os.makedirs(data_dir, exist_ok=True)

unzip_file(elephant_zip, os.path.join(data_dir, 'elephant'))
unzip_file(tiger_zip, os.path.join(data_dir, 'tiger'))
unzip_file(others_zip, os.path.join(data_dir, 'others'))

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

# %%
datagen


# %%

# Generate data for training and validation
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')


# %%
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# %%
base_model


# %%
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# %%
model = Model(inputs=base_model.input, outputs=predictions)

# %%
for layer in base_model.layers:
    layer.trainable = False

# %%
base_model.layers

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# %%
model.save('animal_classification_model.h5')

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)

# Print the validation accuracy
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))

# %%
from tensorflow.keras.models import load_model
from keras.preprocessing import image
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


# %%
# Sample image path change this to a path of a sample image you want to try :)
sample_image_path = 'anushka.jpg'

# Make prediction
prediction = predict_image(sample_image_path)
print("Prediction:", prediction)

# %%



