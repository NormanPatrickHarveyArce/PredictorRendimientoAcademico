def build_features(df):
    df["ratio_tareas"] = df["tareas_entregadas"] / df["tareas_totales"]

    df["engagement_score"] = (
        df["total_accesos_lms"] * 0.4 +
        df["participaciones_foro"] * 0.3 +
        df["tiempo_total_conexion_min"] * 0.3
    )

    df["ratio_asistencia"] = df["asistencias"] / df["clases_totales"]

    return df
