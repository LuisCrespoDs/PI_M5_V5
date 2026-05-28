import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path=None):
    """
    Carga el dataset desde un archivo CSV.

    Si no se recibe una ruta, busca el archivo Base_de_datos.csv
    en la carpeta principal del proyecto.
    """

    if path is None:
        ruta_actual = Path(__file__).resolve()
        ruta_proyecto = ruta_actual.parent.parent
        path = ruta_proyecto / "Base_de_datos.csv"

    df = pd.read_csv(path)
    return df


def build_features(df, target="Pago_atiempo"):
    """
    Separa las variables predictoras de la variable objetivo.
    También identifica variables numéricas y categóricas.
    """

    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe en el dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """
    Crea el preprocesador usando ColumnTransformer.

    Variables numéricas:
    - Imputación con mediana
    - Escalado con StandardScaler

    Variables categóricas:
    - Imputación con la moda
    - Codificación con OneHotEncoder
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def get_train_test_data(path=None, target="Pago_atiempo", test_size=0.2, random_state=42):
    """
    Carga los datos, separa X e y, construye el preprocesador
    y divide el dataset en entrenamiento y prueba.
    """

    df = load_data(path)

    X, y, numeric_features, categorical_features = build_features(df, target=target)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = get_train_test_data()

    print("Ingeniería de características completada.")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("Preprocesador creado:", preprocessor)