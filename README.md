# FBN1-positive CEL Prediction Calculator

This repository contains a Streamlit research prototype for estimating the model-predicted probability of FBN1-positive congenital ectopia lentis using routine blood test parameters.

## Overview

Congenital ectopia lentis (CEL) is commonly associated with pathogenic variants in FBN1. This web application demonstrates a five-feature LightGBM model developed for research purposes. The model uses routinely available blood test parameters to generate a model-derived prediction score for FBN1-positive CEL.

## Required inputs

The calculator requires five laboratory parameters:

- APTT: activated partial thromboplastin time
- Thrombin time
- PDW: platelet distribution width
- Triglycerides
- Platelet count

## Model

The application uses a saved LightGBM Top5 model bundle:

```text
fbn1_cel_lgbm_top5_deployment_full_bundle.joblib