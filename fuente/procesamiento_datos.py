import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def preprocess(df):
    numeric_cols = [
        "edad",
        "nota_primer_parcial",
        "nota_segundo_parcial",
        "promedio_previo",
        "total_accesos_lms",
        "tiempo_total_conexion_min",
        "tareas_entregadas",
        "tareas_totales",
        "participaciones_foro",
        "retroalimentaciones_docente"
    ]

    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
