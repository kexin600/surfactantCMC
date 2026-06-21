# surfactantCMC
Code for predicting the critical micelle concentration (CMC) of surfactants

The main code used in this study is provided in the "code" folder. Specifically, the RFECV script for feature screening is available as "RFECV.ipynb"; the code for building conventional machine learning models under default hyperparameters is provided in "models based on default hyperparameters.ipynb"; and the hyperparameter optimization for ensemble methods is implemented in "Superparametric optimization for catboost.ipynb". For models based on different descriptor types or algorithms, the corresponding scripts may require minor modifications according to the naming conventions; these are not exhaustively listed here.

The primary datasets used in this work are available in the "data" folder. The feature‑selected dataset for the Mordred‑based CatBoost model is provided as "mordred-13.CatBoost best.xlsx", and that for the Morgan fingerprint‑based LightGBM model is provided as "morgan-14.LightGBM best.xlsx".

The GUI program developed in this study is available in the folder "GUI for AFP visualization", which deploys the trained optimal Attentive FP model and enables interactive visualization for single‑molecule interpretation.
