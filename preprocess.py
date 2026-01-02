# preprocess.py
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image):
    """
    Preprocess input PIL image for ResNet50
    """
    # Resize to ResNet50 input size
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    return img_array
