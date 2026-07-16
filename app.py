# -*- coding: utf-8 -*-

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="FBN1-positive CEL Prediction Calculator",
    page_icon="🧬",
    layout="centered"
)


# =============================================================================
# Page styling
# =============================================================================

st.markdown(
    """
    <style>
    /* Overall page width and spacing */
    .block-container {
        max-width: 860px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .title-text {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #1f2937;
    }

    /* Subtitle */
    .subtitle-text {
        text-align: center;
        color: #5f6b7a;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
    }

    /* Section title */
    h3 {
        text-align: center;
        margin-top: 1.4rem;
        margin-bottom: 1rem;
    }

    /* Alert box text */
    div[data-testid="stAlert"] {
        text-align: center;
    }

    /* Button */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 10px;
        height: 2.8rem;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Result card */
    .result-card {
        text-align: center;
        background-color: #f7f9fc;
        padding: 1.35rem 1.45rem;
        border-radius: 14px;
        border: 1px solid #e1e7ef;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    }

    .result-label {
        font-size: 1.02rem;
        color: #4b5563;
        margin-bottom: 0.35rem;
    }

    .result-value {
        font-size: 2.9rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }

    .threshold-text {
        font-size: 0.9rem;
        color: #6b7280;
    }

    .small-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }

    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# Load model bundle
# =============================================================================

@st.cache_resource
def load_bundle():
    bundle_path = Path("fbn1_cel_lgbm_top5_deployment_full_bundle.joblib")
    if not bundle_path.exists():
        st.error(
            "Model bundle file was not found. Please make sure "
            "`fbn1_cel_lgbm_top5_deployment_full_bundle.joblib` is in the same folder as app.py."
        )
        st.stop()
    return joblib.load(bundle_path)


bundle = load_bundle()

model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
features = bundle["features"]
threshold = bundle.get("threshold", 0.5)


# =============================================================================
# Header
# =============================================================================

st.markdown(
    '<div class="title-text">FBN1-positive CEL Prediction Calculator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">'
    'A research prototype based on a five-feature LightGBM model using routine blood test parameters.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "This tool is intended for research and clinical decision-support exploration only. "
    "It does not replace genetic testing or professional clinical judgment."
)


# =============================================================================
# Input section
# =============================================================================

st.markdown("### Input laboratory parameters")

col1, col2 = st.columns(2)

with col1:
    aptt = st.number_input(
        "APTT, activated partial thromboplastin time (s)",
        min_value=0.0,
        max_value=120.0,
        value=32.5,
        step=0.1
    )

    pdw = st.number_input(
        "PDW, platelet distribution width",
        min_value=0.0,
        max_value=100.0,
        value=12.4,
        step=0.1
    )

    platelet_count = st.number_input(
        "Platelet count, PLT (10⁹/L)",
        min_value=0.0,
        max_value=1000.0,
        value=260.0,
        step=1.0
    )

with col2:
    thrombin_time = st.number_input(
        "Thrombin time, TT (s)",
        min_value=0.0,
        max_value=120.0,
        value=16.8,
        step=0.1
    )

    triglycerides = st.number_input(
        "Triglycerides, TG (mmol/L)",
        min_value=0.0,
        max_value=30.0,
        value=1.15,
        step=0.01
    )


# =============================================================================
# Prediction
# =============================================================================

if st.button("Predict probability", type="primary"):

    raw_input = pd.DataFrame([{
        "aptt": aptt,
        "thrombin_time": thrombin_time,
        "pdw": pdw,
        "triglycerides": triglycerides,
        "platelet_count": platelet_count
    }])

    # Keep the exact feature order used by the saved model bundle
    raw_input = raw_input[features]

    X_imp = pd.DataFrame(
        imputer.transform(raw_input),
        columns=features
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X_imp),
        columns=features
    )

    probability = model.predict_proba(X_scaled)[0, 1]
    predicted_class = int(probability >= threshold)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                Model-predicted probability of FBN1-positive CEL
            </div>
            <div class="result-value">
                {probability:.1%}
            </div>
            <div class="threshold-text">
                Model decision threshold: {threshold:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if predicted_class == 1:
        st.warning(
            "Model classification: higher predicted probability of FBN1-positive CEL."
        )
    else:
        st.success(
            "Model classification: lower predicted probability of FBN1-positive CEL."
        )

    st.markdown(
        """
        <div class="small-note">
        This value is a model-derived prediction score and should not be interpreted as a definitive genetic diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer-note">
    Required inputs: APTT, thrombin time, PDW, triglycerides, and platelet count. 
    The prediction is generated by a saved LightGBM model trained for research purposes.
    </div>
    """,
    unsafe_allow_html=True
)