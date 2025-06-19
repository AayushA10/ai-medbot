# scripts/generate_dummy_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

# Dummy CNN model for 4 classes
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.save("model/skin_model.h5")
print("✅ Dummy model saved to model/skin_model.h5")
