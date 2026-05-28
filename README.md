# # PI_M5_V5 - Proyecto Integrador MLOps

## Contexto del proyecto

Este proyecto integrador se desarrolla en el contexto de una empresa financiera que trabaja con información histórica de créditos, clientes y comportamiento de pago. La organización busca fortalecer su proceso de toma de decisiones mediante el uso de analítica de datos, aprendizaje automático y prácticas de MLOps.

Dentro de este escenario, asumo el rol de **Científico de Datos / Analista de Datos Junior Advanced**, con la responsabilidad de construir una solución capaz de analizar datos históricos de préstamos, entrenar modelos predictivos, monitorear cambios en los datos y disponibilizar el modelo mediante una API.

El objetivo principal del proyecto es desarrollar un flujo completo de trabajo que permita predecir si un cliente pagará un crédito a tiempo, utilizando variables relacionadas con el perfil del cliente, características del préstamo, historial crediticio y comportamiento financiero.

---

## Objetivo general

Desarrollar un modelo predictivo supervisado para anticipar el comportamiento de pago de nuevos usuarios, integrando etapas de carga de datos, análisis exploratorio, ingeniería de características, entrenamiento de modelos, monitoreo de data drift y despliegue mediante una API.

---

## Caso de negocio

La empresa financiera necesita anticipar el riesgo asociado a nuevos créditos. Una predicción temprana sobre si un cliente pagará a tiempo puede ayudar a mejorar la toma de decisiones, optimizar procesos de evaluación crediticia y reducir posibles pérdidas asociadas a incumplimientos.

La variable objetivo utilizada en este proyecto es:

```text
Pago_atiempo
```

Esta variable permite identificar si un crédito fue pagado dentro del tiempo esperado. A partir de ella, se entrenan modelos de clasificación supervisada para apoyar la predicción del comportamiento de pago.

---

## Estructura del proyecto

```text
PI_M5_V5/
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
│
├── models/
│   └── best_model.joblib
│
├── app.py
├── Base_de_datos.csv
├── Base_de_datos.xlsx
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

---

## Avance 1: Carga de datos y análisis exploratorio

En el primer avance se creó la estructura inicial del repositorio y se trabajó con los notebooks principales del proyecto.

### Actividades realizadas

* Creación del repositorio en GitHub.
* Configuración de ramas de trabajo.
* Creación de la estructura de carpetas solicitada.
* Carga del archivo original `Base_de_datos.xlsx`.
* Generación del archivo `Base_de_datos.csv`.
* Exploración inicial del dataset.
* Revisión de dimensiones, columnas, tipos de datos, valores nulos y duplicados.
* Análisis univariable, bivariable y multivariable.
* Interpretación del negocio a partir de la variable objetivo `Pago_atiempo`.

### Archivos principales

```text
src/Cargar_datos.ipynb
src/comprension_eda.ipynb
```

---

## Avance 2: Ingeniería de características y entrenamiento de modelos

En el segundo avance se construyó el flujo de preparación de datos y se entrenaron los primeros modelos supervisados.

### Ingeniería de características

Se desarrolló el archivo `ft_engineering.py`, encargado de preparar los datos para el entrenamiento. El proceso incluye:

* Separación entre variables predictoras y variable objetivo.
* Identificación de variables numéricas y categóricas.
* División del dataset en entrenamiento y prueba.
* Creación de pipelines de preprocesamiento.
* Uso de `ColumnTransformer` para aplicar transformaciones específicas por tipo de variable.

Para variables numéricas se aplicó imputación con mediana y escalado. Para variables categóricas se aplicó imputación con la moda y codificación mediante OneHotEncoder.

### Entrenamiento y evaluación de modelos

Se desarrolló el archivo `model_training_evaluation.py`, donde se entrenaron modelos de clasificación supervisada:

* Logistic Regression
* Decision Tree
* Random Forest

Los modelos fueron evaluados utilizando métricas como:

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión

El mejor modelo fue seleccionado según su desempeño y guardado como:

```text
models/best_model.joblib
```

---

## Avance 3: Monitoreo y detección de Data Drift

En el tercer avance se implementó un proceso de monitoreo para detectar cambios en la distribución de los datos.

### Actividades realizadas

* Creación del archivo `model_monitoring.py`.
* División del dataset en muestra de referencia y muestra actual simulada.
* Cálculo de métricas estadísticas para detectar data drift.
* Desarrollo de una aplicación en Streamlit para visualizar resultados.
* Generación de alertas visuales cuando se detecta drift.

### Métricas utilizadas

Para variables numéricas:

* KS Test
* PSI
* Jensen-Shannon divergence

Para variables categóricas:

* Chi-cuadrado

Durante el monitoreo se detectó posible drift en la variable `tipo_laboral`, lo que indica un cambio en la distribución de esa característica entre la muestra de referencia y la muestra actual simulada. Este hallazgo es relevante porque cambios en el perfil laboral de los clientes podrían afectar el rendimiento futuro del modelo.

### Aplicación Streamlit

La aplicación permite visualizar:

* Cantidad de registros de referencia.
* Cantidad de registros actuales.
* Número de variables con drift.
* Tabla de métricas por variable.
* Alertas de riesgo.
* Interpretación del proceso.

Archivo principal:

```text
app.py
```

---

## Avance 4: Despliegue mediante API y Docker

En el cuarto avance se disponibilizó el modelo entrenado mediante una API y se preparó el proyecto para ser ejecutado dentro de un contenedor Docker.

### API con FastAPI

Se desarrolló el archivo `model_deploy.py`, que carga el modelo entrenado desde:

```text
models/best_model.joblib
```

La API incluye los siguientes endpoints:

```text
GET /
POST /predict
POST /predict_batch
```

El endpoint `/predict` permite enviar los datos de un cliente y recibir una predicción sobre si pagará o no a tiempo. El endpoint `/predict_batch` permite realizar predicciones para múltiples registros.

La documentación automática de la API puede visualizarse en:

```text
http://127.0.0.1:8000/docs
```

### Docker

Se creó un `Dockerfile` para definir la imagen del proyecto. Esta imagen contiene:

* Código fuente.
* Dependencias del proyecto.
* Modelo entrenado.
* Servidor de aplicación con Uvicorn.
* API desarrollada con FastAPI.

También se creó el archivo `.dockerignore` para excluir archivos innecesarios durante la construcción de la imagen.

---

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Streamlit
* FastAPI
* Uvicorn
* Joblib
* Docker
* Git
* GitHub

---

## Instalación de dependencias

Para instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

---

## Ejecución del proyecto

### Ejecutar ingeniería de características

```bash
python src/ft_engineering.py
```

### Entrenar y evaluar modelos

```bash
python src/model_training_evaluation.py
```

### Ejecutar monitoreo de data drift

```bash
python src/model_monitoring.py
```

### Ejecutar aplicación Streamlit

```bash
streamlit run app.py
```

### Ejecutar API con FastAPI

```bash
uvicorn src.model_deploy:app --reload
```

Luego abrir en el navegador:

```text
http://127.0.0.1:8000/docs
```

---

## Ejecución con Docker

Para construir la imagen Docker:

```bash
docker build -t pi-m5-v5-api .
```

Para ejecutar el contenedor:

```bash
docker run -p 8000:8000 pi-m5-v5-api
```

Luego abrir:

```text
http://127.0.0.1:8000/docs
```

---

## Principales hallazgos

Durante el análisis exploratorio se identificaron variables relacionadas con el perfil del cliente, el préstamo, el historial crediticio y los saldos asociados. Estas variables permitieron construir un modelo supervisado para predecir el comportamiento de pago.

En la etapa de modelado, los modelos Decision Tree y Random Forest obtuvieron un desempeño alto. Este resultado debe interpretarse con cautela, ya que algunas variables financieras podrían estar muy relacionadas con la variable objetivo. Por ello, en un contexto productivo sería importante revisar posibles fugas de información antes de utilizar el modelo en decisiones reales.

En la etapa de monitoreo se detectó posible drift en la variable `tipo_laboral`, lo que sugiere que el perfil laboral de los clientes puede cambiar con el tiempo. Este tipo de monitoreo es importante porque permite anticipar cuándo un modelo podría perder rendimiento y requerir revisión o reentrenamiento.

---

## Conclusión

Este proyecto integra un flujo completo de trabajo MLOps aplicado a un caso financiero. Se trabajó desde la carga y comprensión inicial de los datos hasta el entrenamiento, evaluación, monitoreo y despliegue del modelo mediante una API.

La solución desarrollada permite no solo construir un modelo predictivo, sino también dejar una base preparada para su uso operativo, monitoreo continuo y futura integración en procesos de negocio.
