import hashlib
import uuid

def anonymize_student_id(student_id, salt):
    value = f"{student_id}{salt}"
    return hashlib.sha256(value.encode()).hexdigest()

def anonymize_dataframe(df, id_column):
    salt = uuid.uuid4().hex  # Salt dinámico por ejecución
    df[id_column] = df[id_column].apply(
        lambda x: anonymize_student_id(x, salt)
    )
    return df
