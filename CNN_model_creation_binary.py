import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


"""
The whole process till here involved selecting confirmed bursters (16) and dippers (21) light curves from the K2 database 
(Rebull et al., 2020). The dataset was then augmented using `augment_func.py` and `create_augmented_data.py`, 
generating 10,000 light curves for each class by applying transformations such as twisting, rotating, flipping, 
and adding noise.

The augmented light curves were then segmented into 10 parts, and each segment was plotted as a 128x128 PNG 
image. This preprocessing step facilitates classification using a CNN-based approach by converting time-series 
data into image format. The resulting dataset was then used for training the model.

I use a binary classifier here: Bursters and Dippers and we get 97% accuracy which is very good!
"""


# Training data set directory
train_data_dir = "./train_data/"  # Parent folder containing bursters_images & dippers_images

#check if directory caught
print("Dataset directories verified!")

# CNN Model Training
IMG_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 20

# Some more data Augmentation & Preprocessing for the CNN 
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_data_dir,  # Use correct parent directory
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Print class mapping
print("Class indices mapping:", train_generator.class_indices) # to check the correct labels of classes: confusion matrix and fo rpredictor_binary.py

val_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Model Definition 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the CNN Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]  # Early stopping added
)

# Save Model After Training
model.save("time_series_classifier_binary.h5")

print("Model successfully trained and saved as time_series_classifier_binary.h5")

print(history.history.keys())

# Plot Training & Validation Loss
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(history.history["loss"], label="Training Loss", color="blue", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training & Validation Loss", fontsize=16, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot Training & Validation Accuracy
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(history.history["accuracy"], label="Training Accuracy", color="green", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange", linestyle="dashed", linewidth=2)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Training & Validation Accuracy", fontsize=16, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("accuracy_plot.png", dpi=300, bbox_inches="tight")
plt.show()



"""
Terminal output: 

Dataset directories verified!
Found 16600 images belonging to 2 classes.
Class indices mapping: {'bursters_images': 0, 'dippers_images': 1}
Found 4150 images belonging to 2 classes.
Epoch 1/20
519/519 [==============================] - 157s 299ms/step - loss: 0.3510 - accuracy: 0.8489 - val_loss: 0.1692 - val_accuracy: 0.9427
Epoch 2/20
519/519 [==============================] - 153s 294ms/step - loss: 0.1434 - accuracy: 0.9519 - val_loss: 0.1185 - val_accuracy: 0.9610
Epoch 3/20
519/519 [==============================] - 150s 290ms/step - loss: 0.0794 - accuracy: 0.9748 - val_loss: 0.1064 - val_accuracy: 0.9617
Epoch 4/20
519/519 [==============================] - 156s 301ms/step - loss: 0.0457 - accuracy: 0.9841 - val_loss: 0.0784 - val_accuracy: 0.9730
Epoch 5/20
519/519 [==============================] - 155s 299ms/step - loss: 0.0315 - accuracy: 0.9885 - val_loss: 0.0838 - val_accuracy: 0.9735
Epoch 6/20
519/519 [==============================] - 202s 390ms/step - loss: 0.0247 - accuracy: 0.9912 - val_loss: 0.0897 - val_accuracy: 0.9728
Epoch 7/20
519/519 [==============================] - 179s 344ms/step - loss: 0.0198 - accuracy: 0.9928 - val_loss: 0.0951 - val_accuracy: 0.9757

Model successfully trained and saved as time_series_classifier_fixed.h5
"""


"""Segment-wise predictions: [5.4980237e-06, 1.521939e-05, 9.953313e-06, 1.1013938e-05, 6.0981574e-06, 1.9813463e-08, 5.552698e-06, 1.7708311e-08, 1.005886e-06, 6.1688996e-05]
Predicted Class: Burster (Confidence: 0.00)

==== Evaluating Model on Validation Set ====
130/130 [==============================] - 14s 106ms/step - loss: 0.0784 - accuracy: 0.9730
Validation Accuracy: 0.9730 | Validation Loss: 0.0784
130/130 [==============================] - 13s 102ms/step

==== Classification Report ====
              precision    recall  f1-score   support

     Dippers       0.97      0.98      0.97      2150
    Bursters       0.98      0.97      0.97      2000

    accuracy                           0.97      4150
   macro avg       0.97      0.97      0.97      4150
weighted avg       0.97      0.97      0.97      4150

"""