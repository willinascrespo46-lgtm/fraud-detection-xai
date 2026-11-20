import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generar_dataset_fraude(n_transacciones: int = 50000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Simular 2,500 clientes con un perfil de gasto habitual
    n_clientes = 2500
    clientes_ids = [f"CLI_{i:05d}" for i in range(1, n_clientes + 1)]
    gasto_habitual_cliente = np.random.lognormal(mean=4.2, sigma=0.6, size=n_clientes) # ~S/ 65 promedio
    perfil_clientes = dict(zip(clientes_ids, gasto_habitual_cliente))
    
    # 2. Asignar clientes y timestamps a lo largo de 90 días
    fecha_inicio = datetime(2026, 8, 1, 0, 0, 0)
    segundos_aleatorios = np.sort(np.random.randint(0, 90 * 24 * 3600, size=n_transacciones))
    timestamps = [fecha_inicio + timedelta(seconds=int(s)) for s in segundos_aleatorios]
    
    tx_clientes = np.random.choice(clientes_ids, size=n_transacciones)
    gasto_base = np.array([perfil_clientes[c] for c in tx_clientes])
    
    # 3. Generar variables transaccionales normales
    horas = np.array([t.hour for t in timestamps])
    categorias = np.random.choice(
        ["Supermercado", "Restaurante", "Transporte", "E-commerce", "Electrónica", "Transferencia", "Viajes"],
        size=n_transacciones,
        p=[0.28, 0.22, 0.18, 0.14, 0.06, 0.08, 0.04]
    )
    canales = np.random.choice(["POS_Fisico", "App_Movil", "Web_Ecommerce"], size=n_transacciones, p=[0.45, 0.35, 0.20])
    
    # Monto normal con variación natural alrededor del gasto base del cliente
    monto_soles = np.round(gasto_base * np.random.lognormal(mean=0.0, sigma=0.45, size=n_transacciones), 2)
    monto_soles = np.clip(monto_soles, 5.0, 15000.0)
    
    # Distancia desde ubicación habitual (km) y antigüedad del dispositivo (días)
    distancia_km = np.round(np.random.exponential(scale=4.5, size=n_transacciones), 1)
    dias_dispositivo = np.random.randint(30, 900, size=n_transacciones)
    intentos_fallidos_previos = np.random.choice([0, 1, 2], size=n_transacciones, p=[0.93, 0.05, 0.02])
    
    # 4. Inyectar patrones de fraude (~1.5% de las transacciones)
    es_fraude = np.zeros(n_transacciones, dtype=int)
    
    # Probabilidad base muy baja + factores de riesgo reales
    prob_fraude = np.full(n_transacciones, 0.002)
    
    # Riesgo A: Compras nocturnas (00:00 a 05:00) en canales digitales
    es_madrugada = (horas >= 0) & (horas <= 5)
    es_digital = np.isin(canales, ["App_Movil", "Web_Ecommerce"])
    prob_fraude += np.where(es_madrugada & es_digital, 0.04, 0.0)
    
    # Riesgo B: Dispositivo nuevo (< 3 días) + categoría de alta liquidez (Electrónica/Transferencia/Viajes)
    dispositivo_nuevo = np.random.rand(n_transacciones) < 0.03
    dias_dispositivo = np.where(dispositivo_nuevo, np.random.randint(0, 3, size=n_transacciones), dias_dispositivo)
    cat_riesgo = np.isin(categorias, ["Electrónica", "Transferencia", "Viajes"])
    prob_fraude += np.where(dispositivo_nuevo & cat_riesgo, 0.18, 0.0)
    
    # Riesgo C: Distancia inusual (> 80 km) + intentos fallidos previos
    viaje_anomalo = np.random.rand(n_transacciones) < 0.025
    distancia_km = np.where(viaje_anomalo, np.round(np.random.uniform(80, 1200, size=n_transacciones), 1), distancia_km)
    prob_fraude += np.where(viaje_anomalo & (intentos_fallidos_previos >= 1), 0.25, 0.0)
    
    # Asignar etiqueta final de fraude y distorsionar monto cuando hay fraude
    es_fraude = (np.random.rand(n_transacciones) < prob_fraude).astype(int)
    
    # En transacciones fraudulentas, los atacantes suelen hacer compras mucho mayores al promedio del cliente
    multiplicador_fraude = np.random.uniform(3.5, 12.0, size=n_transacciones)
    monto_soles = np.where(
        es_fraude == 1,
        np.round(np.clip(gasto_base * multiplicador_fraude, 250.0, 25000.0), 2),
        monto_soles
    )
    
    df = pd.DataFrame({
        "tx_id": [f"TX_{i:06d}" for i in range(1, n_transacciones + 1)],
        "cliente_id": tx_clientes,
        "timestamp": timestamps,
        "monto_soles": monto_soles,
        "categoria_comercio": categorias,
        "canal": canales,
        "distancia_km": distancia_km,
        "dias_dispositivo": dias_dispositivo,
        "intentos_fallidos_previos": intentos_fallidos_previos,
        "es_fraude": es_fraude
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generar_dataset_fraude()
    ruta_salida = "data/transacciones_banco.csv"
    df.to_csv(ruta_salida, index=False)
    tasa = df["es_fraude"].mean() * 100
    print(f"Dataset generado en {ruta_salida}: {len(df):,} filas | Tasa de fraude: {tasa:.2f}%")