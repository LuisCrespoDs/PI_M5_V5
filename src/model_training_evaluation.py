import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ft_engineering import get_train_test_data


def summarize_classification(model_name, y_true, y_pred):
    """
    Resume las métricas principales de un modelo de clasificación.
    """

    summary = {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

    return summary


def build_model(model, preprocessor):
    """
    Construye un pipeline completo con preprocesamiento y modelo.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def train_and_evaluate_models():
    """
    Entrena y evalúa diferentes modelos supervisados de clasificación.
    Guarda el mejor modelo según F1-score.
    """

    X_train, X_test, y_train, y_test, preprocessor = get_train_test_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        print("=" * 80)
        print(f"Entrenando modelo: {model_name}")

        pipeline = build_model(model, preprocessor)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        summary = summarize_classification(model_name, y_test, y_pred)
        results.append(summary)

        trained_pipelines[model_name] = pipeline

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    print("\nResumen comparativo de modelos:")
    print(results_df)

    best_model_name = results_df.iloc[0]["modelo"]
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\nMejor modelo según F1-score: {best_model_name}")

    project_path = Path(__file__).resolve().parent.parent
    models_path = project_path / "models"
    models_path.mkdir(exist_ok=True)

    model_file = models_path / "best_model.joblib"
    joblib.dump(best_pipeline, model_file)

    print(f"Modelo guardado correctamente en: {model_file}")

    return results_df


if __name__ == "__main__":
    results = train_and_evaluate_models()