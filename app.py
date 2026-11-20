import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="GuardianPay | Detección de Fraude XAI", layout="wide")

@st.cache_resource
def cargar_artefactos():
    return joblib.load("models/guardianpay_model.joblib")

try:
    artefactos = cargar_artefactos()
except FileNotFoundError:
    st.error("No se encontró el modelo entrenado. Ejecuta primero `python src/train.py`.")
    st.stop()

pipeline = artefactos["pipeline"]
explainer = artefactos["explainer"]
feature_names = artefactos["feature_names"]
umbral_optimo = artefactos["umbral_optimo"]
metricas = artefactos["metricas"]

st.title("GuardianPay — Centro de Monitoreo de Fraude Bancario (XAI)")
st.markdown(
    f"**Métricas del Modelo (Validación Temporal):** PR-AUC: `{metricas['pr_auc']}` | "
    f"ROC-AUC: `{metricas['roc_auc']}` | Umbral Óptimo Sugerido: `{umbral_optimo:.2f}`"
)
st.divider()

col_input, col_resultado = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("Simulador de Transacción Entrante")
    
    monto_soles = st.number_input("Monto de la operación (S/)", min_value=5.0, max_value=25000.0, value=1850.0, step=50.0)
    gasto_promedio = st.number_input("Gasto promedio histórico del cliente (S/)", min_value=10.0, max_value=5000.0, value=120.0, step=10.0)
    hora_dia = st.slider("Hora del día (0 - 23h)", min_value=0, max_value=23, value=3)
    categoria = st.selectbox("Categoría de comercio", ["Electrónica", "Transferencia", "Viajes", "E-commerce", "Supermercado", "Restaurante", "Transporte"])
    canal = st.selectbox("Canal de operación", ["Web_Ecommerce", "App_Movil", "POS_Fisico"])
    distancia_km = st.slider("Distancia desde ubicación habitual (km)", min_value=0.0, max_value=1200.0, value=140.0, step=5.0)
    dias_dispositivo = st.slider("Antigüedad del dispositivo (días)", min_value=0, max_value=900, value=1, step=1)
    intentos_fallidos = st.selectbox("Intentos fallidos previos", [0, 1, 2], index=1)
    minutos_ultima_tx = st.number_input("Minutos desde la última transacción", min_value=0.5, max_value=9999.0, value=4.0, step=1.0)
    tx_previas = st.number_input("Cantidad de transacciones previas del cliente", min_value=0, max_value=500, value=18, step=1)
    
    umbral_usuario = st.slider("Umbral de bloqueo del analista", min_value=0.05, max_value=0.95, value=round(umbral_optimo, 2), step=0.01)

ratio_monto = round(monto_soles / max(gasto_promedio, 1.0), 2)

datos_tx = pd.DataFrame([{
    "hora_dia": hora_dia,
    "monto_soles": monto_soles,
    "gasto_promedio_historico": gasto_promedio,
    "ratio_monto_vs_promedio": ratio_monto,
    "tx_previas_cliente": tx_previas,
    "minutos_desde_ultima_tx": minutos_ultima_tx,
    "distancia_km": distancia_km,
    "dias_dispositivo": dias_dispositivo,
    "intentos_fallidos_previos": intentos_fallidos,
    "categoria_comercio": categoria,
    "canal": canal,
}])

prob_fraude = float(pipeline.predict_proba(datos_tx)[0, 1])
es_alerta = prob_fraude >= umbral_usuario

# Calcular valores SHAP para esta transacción individual
X_proc = pipeline.named_steps["preprocesador"].transform(datos_tx)
shap_vals = explainer.shap_values(X_proc)[0]

df_shap = pd.DataFrame({
    "Variable": feature_names,
    "Impacto_SHAP": shap_vals
}).sort_values("Impacto_SHAP", key=abs, ascending=True).tail(8)

with col_resultado:
    st.subheader("Evaluación de Riesgo en Tiempo Real")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad de Fraude", f"{prob_fraude * 100:.1f}%")
    m2.metric("Ratio vs Gasto Habitual", f"{ratio_monto}x")
    m3.metric("Decisión del Motor", "BLOQUEAR" if es_alerta else "APROBAR")

    if es_alerta:
        st.error("ALERTA CRÍTICA: Operación retenida por alto riesgo de fraude. Requiere validación del cliente.")
    else:
        st.success("OPERACIÓN SEGURA: Patrón coherente con el perfil histórico del cliente.")

    st.markdown("#### Explicabilidad de la Decisión (Top Factores SHAP)")
    st.caption("Barras rojas aumentan el riesgo de fraude; barras azules reducen el riesgo.")

    colores = ["#EF4444" if v > 0 else "#3B82F6" for v in df_shap["Impacto_SHAP"]]
    fig = go.Figure(go.Bar(
        x=df_shap["Impacto_SHAP"],
        y=df_shap["Variable"],
        orientation="h",
        marker_color=colores
    ))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Contribución al Riesgo (Log-Odds SHAP)")
    st.plotly_chart(fig, use_container_width=True)