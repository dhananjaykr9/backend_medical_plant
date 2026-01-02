# shap_utils.py
# ============================================================
# SHAP Explainability for QPSO-selected features + SVM
# ============================================================

import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# ------------------------------------------------------------
# Generate SHAP explanation
# ------------------------------------------------------------
def generate_shap(features):
    """
    features : numpy array (1, N_features) after QPSO + scaling

    returns : Base64 encoded SHAP image
    """

    # --- Dummy background (required by SHAP)
    # In practice, this should come from training features
    background = np.zeros((10, features.shape[1]))

    # --- KernelExplainer for SVM
    explainer = shap.KernelExplainer(
        lambda x: x,   # identity (feature-level explanation)
        background
    )

    shap_values = explainer.shap_values(features, nsamples=50)

    # --- Plot SHAP values
    plt.figure(figsize=(6, 3))
    shap.summary_plot(
        shap_values,
        features,
        show=False,
        plot_type="bar"
    )

    # --- Convert plot to Base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()

    buf.seek(0)
    shap_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return shap_base64
