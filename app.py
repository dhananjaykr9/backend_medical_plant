# ============================================================
# Medicinal Plant Prediction API
# ResNet50 (TF 2.19 .keras) + QPSO + SVM
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

from preprocess import preprocess_image
from plant_info import get_plant_info


# =====================================================
# MODEL PATHS (already present in repo)
# =====================================================
BASE_MODEL_DIR = "models"

MODEL_PATHS = {
    "resnet": os.path.join(BASE_MODEL_DIR, "resnet_finetuned_tf219.keras"),
    "svm": os.path.join(BASE_MODEL_DIR, "qpso_svm_model_finetuned.pkl"),
    "scaler": os.path.join(BASE_MODEL_DIR, "qpso_scaler_finetuned.pkl"),
    "indices": os.path.join(BASE_MODEL_DIR, "selected_indices_finetuned.npy"),
    "classes": os.path.join(BASE_MODEL_DIR, "class_names.npy"),
}

# Fail fast
for k, p in MODEL_PATHS.items():
    if not os.path.exists(p):
        raise RuntimeError(f"Missing model file: {p}")

print("🔄 Loading models...")

# =====================================================
# LOAD RESNET (TF 2.19 SAFE)
# =====================================================
resnet_model = tf.keras.models.load_model(
    MODEL_PATHS["resnet"],
    compile=False
)
resnet_model.trainable = False

def build_feature_extractor(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            return Model(model.input, layer.output)
    return Model(model.input, model.layers[-2].output)

feature_extractor = build_feature_extractor(resnet_model)

# =====================================================
# LOAD QPSO + SVM
# =====================================================
svm_model = joblib.load(MODEL_PATHS["svm"])
scaler = joblib.load(MODEL_PATHS["scaler"])
selected_indices = np.load(MODEL_PATHS["indices"])
class_names = np.load(MODEL_PATHS["classes"], allow_pickle=True).tolist()

print("✅ All models loaded successfully.")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="Medicinal Plant Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = preprocess_image(image)

    features = feature_extractor.predict(img, verbose=0)
    features = features[:, selected_indices]
    features = scaler.transform(features)

    probs = svm_model.predict_proba(features)[0]
    idx = int(np.argmax(probs))

    return {
        "plant_name": class_names[idx],
        "confidence": float(probs[idx]),
        "description": get_plant_info(class_names[idx]),
    }

@app.get("/")
def health():
    return {"status": "API running"}
