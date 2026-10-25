# -*- coding: utf-8 -*-
from pathlib import Path
import __main__

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_compat import TopKGradientBoostingSelector

# The serialized model refers to this custom class as __main__.TopKGradientBoostingSelector.
__main__.TopKGradientBoostingSelector = TopKGradientBoostingSelector

st.set_page_config(
    page_title="FBN1-related CEL Screening Calculator",
    page_icon="🧬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 900px; padding-top: 1.8rem; padding-bottom: 2rem; }
    .title-text { text-align: center; font-size: 1.9rem; font-weight: 700; margin-bottom: .3rem; }
    .subtitle-text { text-align: center; color: #64748b; font-size: .98rem; margin-bottom: 1rem; }
    .score-card { border: 1px solid #dbe4ee; border-radius: 14px; padding: 1.2rem; text-align: center; background: #f8fafc; margin: .5rem 0 1rem; }
    .score-label { color: #475569; font-size: .95rem; margin-bottom: .25rem; }
    .score-value { color: #0f172a; font-size: 2.6rem; line-height: 1.2; font-weight: 750; }
    .score-foot { color: #64748b; font-size: .86rem; margin-top: .45rem; }
    div.stButton > button { width: 100%; min-height: 2.8rem; font-weight: 600; border-radius: 9px; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_bundle():
    path = Path(__file__).resolve().parent / "locked_FBN1_XGBoost_pipeline.joblib"
    if not path.exists():
        st.error(f"Model file not found: {path.name}. Keep it in the same folder as app.py.")
        st.stop()
    return joblib.load(path)

bundle = load_bundle()
pipeline = bundle["pipeline"]
candidate_features = list(bundle["candidate_features"])
selected_features = list(bundle["selected_features"])
threshold = float(bundle["operating_threshold"])
selected_k = int(bundle.get("selected_k", len(selected_features)))
model_family = bundle.get("model_family", "XGBoost")

# Concise adult reference guides for orientation only. Actual lab reference
# intervals and units may vary by laboratory, method, age, and sex.
FEATURE_INFO = {
    "urea": ("Urea", "mmol/L · adult guide ~2.5–7.1"),
    "inr": ("INR", "unitless · typical guide ~0.8–1.2"),
    "triglycerides": ("Triglycerides", "mmol/L · fasting desirable <1.7"),
    "mpv": ("MPV", "fL · approximate guide 9–13"),
    "plateletcrit": ("Plateletcrit", "% · approximate guide 0.11–0.28"),
    "mchc": ("MCHC", "g/L · approximate guide 316–354"),
    "neutrophil_abs": ("Neutrophil count", "×10⁹/L · approximate guide 1.8–6.3"),
    "alt": ("ALT", "U/L · adult guide often <35–50"),
    "albumin": ("Albumin", "g/L · approximate guide 35–50"),
    "mch": ("MCH", "pg · approximate guide 27–34"),
    "glucose": ("Glucose", "mmol/L · fasting guide ~4.1–5.9"),
    "wbc_count": ("WBC count", "×10⁹/L · approximate guide 3.5–9.5"),
    "calcium": ("Calcium", "mmol/L · approximate guide 2.10–2.55"),
    "prothrombin_time": ("Prothrombin time", "s · approximate guide 8.8–13.8"),
    "total_protein": ("Total protein", "g/L · approximate guide 63–82"),
    "hematocrit": ("Hematocrit", "% · enter e.g. 48.6, not 0.486"),
    "monocyte_pct": ("Monocytes (%)", "% · approximate guide 3–10"),
    "basophil_abs": ("Basophil count", "×10⁹/L · approximate guide 0–0.06"),
    "basophil_pct": ("Basophils (%)", "% · approximate guide 0–1"),
    "fibrinogen": ("Fibrinogen", "g/L · approximate guide 2–5"),
    "platelet_count": ("Platelet count", "×10⁹/L · approximate guide 125–350"),
    "rbc_count": ("RBC count", "×10¹²/L · adult guide varies by sex, ~3.8–5.8"),
    "neutrophil_pct": ("Neutrophils (%)", "% · approximate guide 40–75"),
}

st.markdown('<div class="title-text">FBN1-related CEL Screening Calculator</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle-text">Locked {model_family} model · {selected_k} laboratory features</div>',
    unsafe_allow_html=True,
)
st.info(
    "Research use only. This model score does not establish a genetic diagnosis "
    "and does not replace genetic testing or clinical judgment."
)

st.markdown("### Laboratory inputs")
st.caption(
    "Enter values in the units shown. Reference intervals are brief adult guides only; "
    "use the units and ranges on the original laboratory report. Leave unavailable values blank."
)

input_values = {}
invalid_fields = []
left, right = st.columns(2)

for i, feature in enumerate(selected_features):
    label, hint = FEATURE_INFO.get(
        feature, (feature.replace("_", " ").title(), "Use the unit shown on the laboratory report")
    )
    col = left if i % 2 == 0 else right
    with col:
        raw = st.text_input(
            label,
            value="",
            placeholder=hint,
            key=f"input_{feature}",
            help=f"Approximate guide: {hint}. This is not a diagnostic reference interval.",
        ).strip()
        if raw == "":
            input_values[feature] = np.nan
        else:
            try:
                input_values[feature] = float(raw)
            except ValueError:
                input_values[feature] = np.nan
                invalid_fields.append(label)

if invalid_fields:
    st.warning("Please enter numeric values or leave blank: " + ", ".join(invalid_fields))

if st.button("Calculate screening score", type="primary"):
    if invalid_fields:
        st.error("Correct the non-numeric inputs before calculating.")
    else:
        # The saved pipeline expects all candidate columns in the original order.
        # Only the selected features are requested from the user; other candidates
        # are left missing and handled by the pipeline's fitted imputer.
        raw_input = pd.DataFrame(
            [{feature: np.nan for feature in candidate_features}],
            columns=candidate_features,
        )
        for feature, value in input_values.items():
            raw_input.loc[0, feature] = value

        probability = float(pipeline.predict_proba(raw_input[candidate_features])[0, 1])
        predicted_class = int(probability >= threshold)

        st.markdown("### Prediction result")
        st.metric(
            label="Model-predicted screening score",
            value=f"{probability:.1%}",
        )
        st.caption(f"Decision threshold: {threshold:.3f}")

        if predicted_class == 1:
            st.warning("Model classification: screen-positive.")
        else:
            st.success("Model classification: screen-negative.")

        st.caption(
            "This is a model-derived score from a matched case-control research cohort, "
            "not an absolute population risk or a definitive FBN1 diagnosis."
        )

with st.expander("Model details"):
    st.write(f"Model: {model_family}")
    st.write(f"Selected features: {selected_k}")
    st.write(f"Decision threshold: {threshold:.3f}")
    st.caption("Reference intervals are for input orientation only and may differ from your laboratory's ranges.")

st.markdown("---")
st.caption("Research prototype · Confirm all units against the original laboratory report.")
