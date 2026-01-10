# ============================================================
# SISTEMA PREDICTIVO DE RENDIMIENTO ACADEMICO UNIVERSITARIO
# Pipeline completo en un solo archivo
# ============================================================

import pandas as pd
import numpy as np
import hashlib
import uuid

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------

df = pd.read_csv("datos_moodle.csv")

print("Datos cargados correctamente")
print(df.head())

# ------------------------------------------------------------
# 2. ANONIMIZACION (Ley 29733)
# ------------------------------------------------------------

def anonymize_id(value, salt):
    return hashlib.sha256(f"{value}{salt}".encode()).hexdigest()

salt = uuid.uuid4().hex
df["id_anonimo"] = df["codigo_estudiante"].apply(lambda x: anonymize_id(x, salt))
df.drop(columns=["codigo_estudiante"], inplace=True)

print("Anonimización aplicada")

# ------------------------------------------------------------
# 3. VARIABLE OBJETIVO (RIESGO ACADEMICO)
# ------------------------------------------------------------

df["riesgo_academico"] = (df["nota_final"] < 11).astype(int)

# ------------------------------------------------------------
# 4. INGENIERIA DE CARACTERISTICAS
# ------------------------------------------------------------

df["ratio_tareas"] = df["tareas_entregadas"] / df["tareas_totales"]
df["ratio_asistencia"] = df["asistencias"] / df["clases_totales"]

df["engagement_score"] = (
    df["total_accesos_lms"] * 0.4 +
    df["participaciones_foro"] * 0.3 +
    df["tiempo_total_conexion_min"] * 0.3
)

# ------------------------------------------------------------
# 5. SELECCION DE VARIABLES
# ------------------------------------------------------------

features = [
    "edad",
    "nota_primer_parcial",
    "nota_segundo_parcial",
    "promedio_previo",
    "total_accesos_lms",
    "tiempo_total_conexion_min",
    "participaciones_foro",
    "ratio_tareas",
    "ratio_asistencia",
    "engagement_score",
    "retroalimentaciones_docente"
]

X = df[features]
y = df["riesgo_academico"]

# ------------------------------------------------------------
# 6. PREPROCESAMIENTO
# ------------------------------------------------------------

imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

# ------------------------------------------------------------
# 7. DIVISION TRAIN / TEST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 8. ENTRENAMIENTO DE MODELOS
# ------------------------------------------------------------

models = {
    "Regresion_Logistica": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Red_Neuronal": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print(f"\nModelo: {name}")
    print(classification_report(y_test, y_pred))

    results.append({
        "Modelo": name,
        "ROC_AUC": auc
    })

# ------------------------------------------------------------
# 9. RESULTADOS COMPARATIVOS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv("resultados_modelos.csv", index=False)

print("\nResultados comparativos guardados")

# ------------------------------------------------------------
# 10. DATASET FINAL PARA POWER BI
# ------------------------------------------------------------

df_powerbi = df.copy()
df_powerbi["probabilidad_riesgo"] = models["Random_Forest"].predict_proba(X)[:, 1]

df_powerbi.to_csv("dataset_powerbi.csv", index=False)

print("Dataset para Power BI generado correctamente")
