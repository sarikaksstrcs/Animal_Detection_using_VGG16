import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define constants
input_shape = (224, 224)
num_classes = 1  # 1 for elephant, 0 for non-elephant
batch_size = 32
epochs = 10

# Set the data directory
data_dir = 'elephant'

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) 

# split
all_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='binary',
    subset='training')  # train
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')  # validation

# Load
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#for binary
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=predictions)

# Freeze 
for layer in base_model.layers:
    layer.trainable = False

# Compile 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train 
model.fit(
    all_generator,
    steps_per_epoch=all_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Save 
model.save('elephant_detection_model.h5')

validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)

# Print the validation accuracy
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))