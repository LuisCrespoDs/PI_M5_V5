import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from model_monitoring import run_monitoring


st.set_page_config(
    page_title="Monitoreo de Data Drift",
    layout="wide"
)

st.title("Monitoreo de Data Drift")
st.write(
    """
    Esta aplicación permite visualizar métricas de monitoreo para detectar posibles cambios
    en la distribución de los datos entre una muestra de referencia y una muestra actual.
    """
)

drift_results, reference, current = run_monitoring()

st.subheader("Resumen general")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Registros de referencia", reference.shape[0])

with col2:
    st.metric("Registros actuales", current.shape[0])

with col3:
    variables_con_drift = (drift_results["drift_detectado"] == "Si").sum()
    st.metric("Variables con drift", variables_con_drift)

st.subheader("Tabla de métricas de drift")
st.dataframe(drift_results, use_container_width=True)

st.subheader("Alertas")

if variables_con_drift > 0:
    st.warning(
        f"Se detectaron {variables_con_drift} variable(s) con posible data drift. "
        "Se recomienda revisar estas variables y evaluar si es necesario reentrenar el modelo."
    )
else:
    st.success("No se detectaron señales relevantes de data drift.")

st.subheader("Variables con drift detectado")

drift_alerts = drift_results[drift_results["drift_detectado"] == "Si"]

if not drift_alerts.empty:
    st.dataframe(drift_alerts, use_container_width=True)
else:
    st.info("No hay variables con drift detectado.")

st.subheader("Interpretación del proceso")

st.write(
    """
    El monitoreo compara una muestra de referencia contra una muestra actual simulada.
    Para variables numéricas se utilizan métricas como KS Test, PSI y Jensen-Shannon.
    Para variables categóricas se utiliza la prueba Chi-cuadrado.

    Estas métricas permiten identificar cambios en la población que podrían afectar
    el desempeño futuro del modelo. Si se detecta drift en variables importantes,
    se recomienda revisar la calidad de los datos, analizar la causa del cambio
    y considerar un posible reentrenamiento del modelo.
    """
)