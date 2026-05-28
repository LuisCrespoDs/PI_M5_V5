import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


def load_data(path=None):
    """
    Carga la base de datos principal.
    """

    if path is None:
        ruta_actual = Path(__file__).resolve()
        ruta_proyecto = ruta_actual.parent.parent
        path = ruta_proyecto / "Base_de_datos.csv"

    return pd.read_csv(path)


def split_reference_current(df, reference_size=0.5, random_state=42):
    """
    Divide el dataset en datos de referencia y datos actuales simulados.
    """

    reference = df.sample(frac=reference_size, random_state=random_state)
    current = df.drop(reference.index)

    return reference, current


def calculate_psi(expected, actual, buckets=10):
    """
    Calcula Population Stability Index para variables numéricas.
    """

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if expected.nunique() <= 1 or actual.nunique() <= 1:
        return np.nan

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return np.nan

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))

    return psi


def calculate_numeric_drift(reference, current, numeric_columns):
    """
    Calcula métricas de drift para variables numéricas:
    - KS test
    - PSI
    - Jensen-Shannon divergence
    """

    results = []

    for column in numeric_columns:
        ref = reference[column].dropna()
        cur = current[column].dropna()

        if ref.empty or cur.empty:
            continue

        ks_stat, ks_pvalue = ks_2samp(ref, cur)
        psi_value = calculate_psi(ref, cur)

        ref_hist, bins = np.histogram(ref, bins=10, density=True)
        cur_hist, _ = np.histogram(cur, bins=bins, density=True)

        ref_hist = ref_hist + 0.0001
        cur_hist = cur_hist + 0.0001

        js_value = jensenshannon(ref_hist, cur_hist)

        results.append({
            "variable": column,
            "tipo": "numerica",
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "psi": round(psi_value, 4) if not np.isnan(psi_value) else np.nan,
            "jensen_shannon": round(js_value, 4),
            "drift_detectado": "Si" if ks_pvalue < 0.05 or (not np.isnan(psi_value) and psi_value > 0.2) else "No"
        })

    return pd.DataFrame(results)


def calculate_categorical_drift(reference, current, categorical_columns):
    """
    Calcula Chi-cuadrado para variables categóricas.
    """

    results = []

    for column in categorical_columns:
        ref_counts = reference[column].fillna("Nulo").value_counts()
        cur_counts = current[column].fillna("Nulo").value_counts()

        categories = list(set(ref_counts.index).union(set(cur_counts.index)))

        ref_values = [ref_counts.get(cat, 0) for cat in categories]
        cur_values = [cur_counts.get(cat, 0) for cat in categories]

        contingency_table = np.array([ref_values, cur_values])

        try:
            chi2, pvalue, _, _ = chi2_contingency(contingency_table)

            results.append({
                "variable": column,
                "tipo": "categorica",
                "chi2_statistic": round(chi2, 4),
                "chi2_pvalue": round(pvalue, 4),
                "drift_detectado": "Si" if pvalue < 0.05 else "No"
            })

        except ValueError:
            continue

    return pd.DataFrame(results)


def run_monitoring():
    """
    Ejecuta el proceso completo de monitoreo y detección de data drift.
    """

    df = load_data()

    reference, current = split_reference_current(df)

    target = "Pago_atiempo"

    X_reference = reference.drop(columns=[target], errors="ignore")
    X_current = current.drop(columns=[target], errors="ignore")

    numeric_columns = X_reference.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X_reference.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_results = calculate_numeric_drift(X_reference, X_current, numeric_columns)
    categorical_results = calculate_categorical_drift(X_reference, X_current, categorical_columns)

    drift_results = pd.concat([numeric_results, categorical_results], ignore_index=True)

    return drift_results, reference, current


if __name__ == "__main__":
    drift_results, reference, current = run_monitoring()

    print("Monitoreo de data drift completado.")
    print("\nTamaño datos de referencia:", reference.shape)
    print("Tamaño datos actuales:", current.shape)
    print("\nResultados de drift:")
    print(drift_results)