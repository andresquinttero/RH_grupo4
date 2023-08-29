import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas

import os

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


# Normalización de datos
def normalize_data(df, numerical_features):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df, scaler

# Codificación de etiquetas para variables categóricas
def label_encode(df, categorical_features):
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        df[feature] = label_encoder.fit_transform(df[feature])
    return df, label_encoder

# Mostrar matriz de confusión
def show_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix

# Obtener importancia de las características para un modelo dado
def feature_importance(model, feature_names):
    importance = model.feature_importances_
    feature_importance_dict = {}
    for i, j in enumerate(importance):
        feature_importance_dict[feature_names[i]] = j
    sorted_features = {k: v for k, v in sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_features

# Guardar el modelo y las transformaciones
def save_transformations_and_model(scaler, label_encoder, model, model_name):
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    joblib.dump(label_encoder, f"{model_name}_label_encoder.pkl")
    joblib.dump(model, f"{model_name}_model.pkl")

# Cargar el modelo y las transformaciones
def load_transformations_and_model(model_name):
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    label_encoder = joblib.load(f"{model_name}_label_encoder.pkl")
    model = joblib.load(f"{model_name}_model.pkl")
    return scaler, label_encoder, model

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
        
        
#NO QUIERE FUNCIONAR
def analyze_correlation(conn, target_variable='Attrition', threshold=0.15):
    # Leer datos desde la base de datos SQL para análisis de correlación
    df = pd.read_sql("SELECT * FROM all_employees", conn)

    # Analizar la correlación
    print(df.columns.tolist())
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df_numeric.corr()
    filter_ = np.abs(corr_matrix[target_variable]) > threshold
    corr_features = corr_matrix.columns[filter_].tolist()

    plt.figure(figsize=(15, 10))
    sns.heatmap(df[corr_features].corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt=".2f")
    plt.title(f"Correlación entre variables (Threshold: {threshold})")
    plt.show()