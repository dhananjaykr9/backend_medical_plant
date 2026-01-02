# gradcam.py
# ============================================================
# Grad-CAM for ResNet50 (Keras)
# ============================================================

import numpy as np
import cv2
import tensorflow as tf
import base64

# ------------------------------------------------------------
# Utility: find last convolutional layer
# ------------------------------------------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# ------------------------------------------------------------
# Generate Grad-CAM
# ------------------------------------------------------------
def generate_gradcam(model, img_array, class_index):
    """
    model       : Keras ResNet model
    img_array   : preprocessed image (1, 224, 224, 3)
    class_index : predicted class index (int)

    returns: Base64 encoded Grad-CAM image
    """

    last_conv_layer_name = find_last_conv_layer(model)

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolution outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Apply ReLU
    heatmap = np.maximum(heatmap, 0)

    # Normalize
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Resize to image size
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original image back (approximate visualization)
    img = img_array[0]
    img = img - img.min()
    img = img / img.max()
    img = np.uint8(255 * img)

    # Overlay heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Encode to Base64
    _, buffer = cv2.imencode(".jpg", superimposed_img)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    return gradcam_base64
