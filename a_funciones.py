import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas
def ejecutar_sql(nombre_archivo, cur):
    try:
        # Mirar si el archivo existe
        if not os.path.exists(nombre_archivo):
            raise FileNotFoundError(f"El archivo {nombre_archivo} no se encuentra")
        
        # Abir el archivo SQL
        with open(nombre_archivo, 'r') as sql_file:
            sql_as_string = sql_file.read()
        
        # Ejecutar el SQL
        cur.executescript(sql_as_string)
        
    except Exception as e:
        # Imprimir el error si lo hay
        print("Error al ejecutar el archivo SQL:", str(e))

# Mostrar matriz de confusión
def show_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix
# Identificar y quitar atípicos
def identify_and_remove_outliers(conn, columns, threshold=2.1):
    # Leer datos desde la base de datos SQL
    df = pd.read_sql("SELECT * FROM all_employees", conn)

    for column in columns:
        Q1 = np.quantile(df[column], 0.25)
        Q3 = np.quantile(df[column], 0.75)
        IQR = Q3 - Q1
        upper = Q3 + threshold * IQR
        lower = Q1 - threshold * IQR

        # Usar SQL para eliminar outliers
        query = f"DELETE FROM all_employees WHERE {column} > {upper} OR {column} < {lower}"
        conn.execute(query)
        conn.commit()
        