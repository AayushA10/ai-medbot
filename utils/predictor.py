# utils/predictor.py

import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model once (assumes you’ll put skin_model.h5 in model/)
model = tf.keras.models.load_model("model/skin_model.h5")

# Label mapping (update as per your actual model)
CLASS_NAMES = ["Eczema", "Psoriasis", "Melanoma", "Ringworm"]

def predict_skin_disease(image: Image.Image):
    image = image.convert("RGB")  # 🔥 Convert RGBA → RGB (3 channels only)
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    top_idx = np.argmax(preds)
    return CLASS_NAMES[top_idx], float(preds[top_idx])
