import os
import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)


def cargar_features_con_sql(ruta_csv: str, ruta_sql: str) -> pd.DataFrame:
    """Carga el CSV crudo y ejecuta la consulta SQL con Window Functions usando DuckDB."""
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE transacciones_banco AS SELECT * FROM read_csv_auto('{ruta_csv}');"
    )
    with open(ruta_sql, "r", encoding="utf-8") as f:
        query = f.read()
    df_features = con.execute(query).df()
    con.close()
    return df_features


def entrenar_modelo_fraude():
    ruta_csv = "data/transacciones_banco.csv"
    ruta_sql = "sql/feature_engineering.sql"

    print("1. Ejecutando ingeniería de variables con SQL (DuckDB)...")
    df = cargar_features_con_sql(ruta_csv, ruta_sql)

    # Ordenar cronológicamente para hacer un split temporal (estándar en banca)
    df = df.sort_values("timestamp").reset_index(drop=True)

    features_numericas = [
        "hora_dia",
        "monto_soles",
        "gasto_promedio_historico",
        "ratio_monto_vs_promedio",
        "tx_previas_cliente",
        "minutos_desde_ultima_tx",
        "distancia_km",
        "dias_dispositivo",
        "intentos_fallidos_previos",
    ]
    features_categoricas = ["categoria_comercio", "canal"]

    X = df[features_numericas + features_categoricas]
    y = df["es_fraude"]

    # Split temporal: 80% pasado para entrenar, 20% futuro para evaluar (sin shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    # Calcular peso de desbalanceo (negativos / positivos) para XGBoost
    ratio_desbalanceo = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"2. Ratio de desbalanceo en train (scale_pos_weight): {ratio_desbalanceo:.2f}")

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", "passthrough", features_numericas),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                features_categoricas,
            ),
        ]
    )

    modelo_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio_desbalanceo,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", modelo_xgb),
        ]
    )

    print("3. Entrenando modelo XGBoost...")
    pipeline.fit(X_train, y_train)

    # Evaluación con métricas reales para datos desbalanceados
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)

    # Encontrar el umbral óptimo que maximiza el F1-Score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    mejor_idx = np.argmax(f1_scores)
    umbral_optimo = float(thresholds[min(mejor_idx, len(thresholds) - 1)])

    y_pred_optimo = (y_probs >= umbral_optimo).astype(int)

    print("\n--- RESULTADOS EN TEST (20% FUTURO) ---")
    print(f"PR-AUC (Métrica principal en fraude): {pr_auc:.4f}")
    print(f"ROC-AUC:                              {roc_auc:.4f}")
    print(f"Umbral óptimo de decisión (F1 max):   {umbral_optimo:.4f}")
    print("\nReporte de Clasificación con Umbral Óptimo:")
    print(classification_report(y_test, y_pred_optimo, target_names=["Legítima", "Fraude"]))

    # Crear explicador SHAP sobre el modelo XGBoost entrenado
    print("4. Construyendo explicador SHAP (TreeExplainer)...")
    X_test_transformado = pipeline.named_steps["preprocesador"].transform(X_test)
    nombres_features = (
        features_numericas
        + list(
            pipeline.named_steps["preprocesador"]
            .named_transformers_["cat"]
            .get_feature_names_out(features_categoricas)
        )
    )
    explainer = shap.TreeExplainer(pipeline.named_steps["clasificador"])

    # Guardar artefactos en disco
    os.makedirs("models", exist_ok=True)
    artefactos = {
        "pipeline": pipeline,
        "explainer": explainer,
        "feature_names": nombres_features,
        "features_numericas": features_numericas,
        "features_categoricas": features_categoricas,
        "umbral_optimo": umbral_optimo,
        "metricas": {"pr_auc": round(pr_auc, 4), "roc_auc": round(roc_auc, 4)},
    }
    joblib.dump(artefactos, "models/guardianpay_model.joblib")
    print("Modelo y explicador SHAP guardados en models/guardianpay_model.joblib")


if __name__ == "__main__":
    entrenar_modelo_fraude()