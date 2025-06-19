import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

model = tf.keras.models.load_model("model/skin_model.h5")

def generate_gradcam(image: Image.Image, class_index: int,
                     last_conv_layer_name: str = "conv2d_1") -> Image.Image:
    # 1. Pre-process
    image = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, 0)        # (1,224,224,3)

    # 2. Target conv layer
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(f"No layer named {last_conv_layer_name}")

    # 3. Build grad-model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == target_layer.name:
            conv_out = x
    preds = x
    grad_model = tf.keras.Model(inputs, [conv_out, preds])

    # 4. Forward + gradients
    with tf.GradientTape() as tape:
        conv_maps, pred = grad_model(img_array)
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_maps)[0].numpy()
    conv_maps = conv_maps[0].numpy()

    # 5. Weighted sum
    weights = grads.mean(axis=(0, 1))
    cam = np.zeros(conv_maps.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_maps[:, :, i]

    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-8
    cam = cv2.resize(cam, (224, 224))

    # 6. Overlay
    heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(np.array(image), 0.6, heat, 0.4, 0)
    return Image.fromarray(blended)
