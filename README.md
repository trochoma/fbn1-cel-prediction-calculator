# FBN1-Related Congenital Ectopia Lentis Screening Calculator

A Streamlit-based research prototype for exploring the use of routine laboratory parameters in screening for FBN1-related congenital ectopia lentis (CEL).

## Model

- **Model family:** XGBoost
- **Selected features:** 23 routine laboratory parameters
- **Model artifact:** `locked_FBN1_XGBoost_pipeline.joblib`
- **Prediction threshold:** Loaded from the locked model artifact

The saved pipeline includes the fitted preprocessing, feature-selection, and prediction steps. The application uses the locked pipeline rather than refitting the model.

## Files

- `app.py`: Streamlit application.
- `model_compat.py`: Compatibility definition for loading the serialized feature selector.
- `locked_FBN1_XGBoost_pipeline.joblib`: Locked model artifact.
- `requirements.txt`: Python dependencies.

## Run Locally

Python 3.10 or a compatible environment is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Intended Use and Limitations

This application is intended for research and exploratory screening only. Its model-derived score should not be interpreted as an absolute population risk or a definitive genetic diagnosis.

The development and temporal-validation cohorts were matched case-control samples. The score and classification threshold should therefore be interpreted in the context of the study design. Genetic testing and professional clinical assessment remain necessary for diagnostic confirmation.

Laboratory values must be entered using the units expected by the model. The displayed reference intervals are approximate guides only and may differ from the reporting laboratory.

## Data Privacy

Do not upload patient-level data, identifiable clinical information, or confidential research records to this repository.

## Disclaimer

This research prototype is not intended to replace professional medical advice, diagnosis, or treatment.
