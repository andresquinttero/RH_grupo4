import sqlite3 as sql #### para bases de datos sql
import pandas as pd # para manejo de datos
from sklearn.feature_selection import SelectKBest # KBest: Seleccione características de acuerdo con las k puntuaciones más altas.
from sklearn.feature_selection import f_classif # cuál de las variables es más importante para la variable objetivo con f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

conn = sql.connect("db_empleados") # Conecatamos la base

df = pd.read_sql("select * from df", conn) # Cargamos los datos

XKbest = df.drop("Attrition", axis=1)  # Características
yKbest = df["Attrition"]  # Variable objetivo

#  Crear modelo de selección f_classif
k_best = SelectKBest(score_func=f_classif, k=5)  # con k= 5, las 5 mejores características

# Ajustar las variables
X_best = k_best.fit_transform(XKbest, yKbest)

# Visualización
# Puntuación de las variables
feature_scores = pd.DataFrame({'Feature': XKbest.columns, 'Score': k_best.scores_})
feature_scores.sort_values(by='Score', ascending=False, inplace=True)

# Variables seleccionadas
selected_featuresKbest = XKbest.columns[k_best.get_support()]

print("Puntuaciones de características:")
print(feature_scores)
print("\nCaracterísticas seleccionadas:")
print(selected_featuresKbest)

# Crea un nuevo DataFrame con las características seleccionadas por KBest
df_variables_kbest = XKbest[selected_featuresKbest].copy()

# Agrega la variable "Attrition" al DataFrame df_variables_kbest
df_variables_kbest['Attrition'] = df['Attrition']

# Con Sequential Feature Selector

Xsfs = df.drop("Attrition", axis=1)  # Características
ysfs = df["Attrition"]  # Variable objetivo

# Crear una instancia de SequentialFeatureSelector
sfs = SequentialFeatureSelector(LogisticRegression(class_weight="balanced", max_iter=1000), 
                                n_features_to_select=5, 
                                direction= "forward",  
                                scoring='f1')

# Ajusta
sfs.fit(Xsfs, ysfs)

# Resultados
selected_featuresSFS = Xsfs.columns[sfs.get_support()]
print("Características seleccionadas:", selected_featuresSFS)

# Crea un nuevo DataFrame con las características seleccionadas por SFS
df_variables_sfs = Xsfs[selected_featuresSFS].copy()
# Agrega la variable "Attrition" al DataFrame df_variables_sfs
df_variables_sfs['Attrition'] = df['Attrition']

print(sorted(selected_featuresKbest))
print(sorted(selected_featuresSFS))

conn = sql.connect("variables_kbest") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.
df_variables_kbest.to_sql("df_variables_kbest", conn, if_exists = "replace", index=False)### Llevar tablas a base de datos

conn1 = sql.connect("variables_sfs") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.
df_variables_sfs.to_sql("df_variables_sfs", conn1, if_exists = "replace", index=False) ### Llevar tablas a base de datos
