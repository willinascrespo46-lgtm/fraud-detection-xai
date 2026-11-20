WITH historial_cliente AS (
    SELECT 
        tx_id,
        cliente_id,
        timestamp,
        monto_soles,
        categoria_comercio,
        canal,
        distancia_km,
        dias_dispositivo,
        intentos_fallidos_previos,
        es_fraude,
        AVG(monto_soles) OVER (
            PARTITION BY cliente_id 
            ORDER BY timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS gasto_promedio_historico,
        COUNT(*) OVER (
            PARTITION BY cliente_id 
            ORDER BY timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS tx_previas_cliente,
        LAG(timestamp) OVER (
            PARTITION BY cliente_id 
            ORDER BY timestamp
        ) AS timestamp_tx_anterior
    FROM transacciones_banco
)
SELECT 
    tx_id,
    cliente_id,
    timestamp,
    EXTRACT(HOUR FROM timestamp) AS hora_dia,
    monto_soles,
    COALESCE(ROUND(gasto_promedio_historico, 2), monto_soles) AS gasto_promedio_historico,
    ROUND(
        monto_soles / NULLIF(COALESCE(gasto_promedio_historico, monto_soles), 0), 
        2
    ) AS ratio_monto_vs_promedio,
    tx_previas_cliente,
    COALESCE(
        ROUND(EXTRACT(EPOCH FROM (timestamp - timestamp_tx_anterior)) / 60.0, 1),
        9999.0
    ) AS minutos_desde_ultima_tx,
    categoria_comercio,
    canal,
    distancia_km,
    dias_dispositivo,
    intentos_fallidos_previos,
    es_fraude
FROM historial_cliente;