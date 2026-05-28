from pathlib import Path
from typing import List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH / "models" / "best_model.joblib"

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="API para disponibilizar el modelo de clasificación entrenado.",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    tipo_credito: int
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str


@app.get("/")
def home():
    return {
        "mensaje": "API funcionando correctamente",
        "endpoint_prediccion": "/predict",
        "documentacion": "/docs"
    }


@app.post("/predict")
def predict(data: PredictionInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df).max()
    else:
        probability = None

    return {
        "prediccion": int(prediction),
        "interpretacion": "Pago a tiempo" if int(prediction) == 1 else "No paga a tiempo",
        "probabilidad": float(probability) if probability is not None else None
    }


@app.post("/predict_batch")
def predict_batch(data: List[PredictionInput]):
    input_df = pd.DataFrame([item.model_dump() for item in data])

    predictions = model.predict(input_df)

    response = []

    for pred in predictions:
        response.append({
            "prediccion": int(pred),
            "interpretacion": "Pago a tiempo" if int(pred) == 1 else "No paga a tiempo"
        })

    return {
        "cantidad_registros": len(response),
        "predicciones": response
    }