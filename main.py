# ============================================================
# Medicinal Plant Prediction API
# Railway-safe version (lazy TensorFlow loading)
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

# =====================================================
# GLOBAL OBJECTS (INITIALIZED AT STARTUP)
# =====================================================
resnet_model = None
feature_extractor = None
svm_model = None
scaler = None
selected_indices = None
class_names = None

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

# =====================================================
# LOAD MODELS ON STARTUP (CRITICAL FIX)
# =====================================================
@app.on_event("startup")
def load_models():
    global resnet_model, feature_extractor
    global svm_model, scaler, selected_indices, class_names

    print("🔄 Loading models on startup...")

    # ---- Check files exist
    for p in MODEL_PATHS.values():
        if not os.path.exists(p):
            raise RuntimeError(f"❌ Missing model file: {p}")

    # ---- Load ResNet (.keras)
    resnet_model = tf.keras.models.load_model(
        MODEL_PATHS["resnet"],
        compile=False
    )
    resnet_model.trainable = False

    # ---- Build feature extractor
    for layer in reversed(resnet_model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            feature_extractor = Model(
                inputs=resnet_model.input,
                outputs=layer.output
            )
            break

    # ---- Load QPSO + SVM assets
    svm_model = joblib.load(MODEL_PATHS["svm"])
    scaler = joblib.load(MODEL_PATHS["scaler"])
    selected_indices = np.load(MODEL_PATHS["indices"])
    class_names = np.load(MODEL_PATHS["classes"], allow_pickle=True).tolist()

    print("✅ All models loaded successfully")

# =====================================================
# HEALTH CHECK (RESPONDS IMMEDIATELY)
# =====================================================
@app.get("/")
def health():
    return {"status": "API running successfully"}

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_array = preprocess_image(image)

    features = feature_extractor.predict(img_array, verbose=0)
    features = features[:, selected_indices]
    features = scaler.transform(features)

    probs = svm_model.predict_proba(features)[0]
    idx = int(np.argmax(probs))

    return {
        "plant_name": class_names[idx],
        "confidence": round(float(probs[idx]), 4),
        "description": get_plant_info(class_names[idx])
    }
