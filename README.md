# GuardianPay — Detección de Fraude Financiero y Explicabilidad (XAI)

Sistema end-to-end de detección de fraude transaccional en tiempo real diseñado para entornos bancarios y fintech en Perú (operaciones en `S/`). Combina **ingeniería de variables comportamentales en SQL (DuckDB)**, modelado para **clases altamente desbalanceadas (XGBoost)**, optimización de umbral mediante **PR-AUC** y **explicabilidad individual con SHAP** desplegada en un centro de monitoreo interactivo.

## Arquitectura del Proyecto

```text
fraud-detection-xai/
├── data/
│   └── generar_transacciones.py   # Generador sintético de 50,000 transacciones (0.89% fraude)
├── sql/
│   └── feature_engineering.sql    # CTEs y Window Functions (gasto histórico y velocidad LAG)
├── src/
│   └── train.py                   # Pipeline DuckDB -> Split Temporal -> XGBoost -> SHAP
├── models/
│   └── guardianpay_model.joblib   # Pipeline entrenado y TreeExplainer serializado
├── app.py                         # Centro de Monitoreo de Fraude en Streamlit + Plotly
└── requirements.txt