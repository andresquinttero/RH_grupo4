import sqlite3 as sql #### para bases de datos sql
import pandas as pd # para manejo de datos
from sklearn.feature_selection import SelectKBest # KBest: Seleccione características de acuerdo con las k puntuaciones más altas.
from sklearn.feature_selection import f_classif # cuál de las variables es más importante para la variable objetivo con f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import  KNeighborsClassifier
import a_funciones
conn = sql.connect("db_empleados") # Conecatamos la base

df = pd.read_sql("select * from df", conn) # Cargamos los datos

XKbest = df.drop("Attrition", axis=1)  # Características
yKbest = df["Attrition"]  # Variable objetivo

#  Crear modelo de selección f_classif
k_best = SelectKBest(score_func=f_classif, k=16)  # con k= 16, las 16 mejores características

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
                                n_features_to_select=16, 
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

print("Variables con KBest:",sorted(selected_featuresKbest))
print("Variables con SFS:",sorted(selected_featuresSFS))


conn = sql.connect("db_empleados")
df_variables_kbest.to_sql("df_variables_kbest", conn, if_exists = "replace", index=False)### Llevar tablas a base de datos
df_variables_sfs.to_sql("df_variables_sfs", conn, if_exists = "replace", index=False) ### Llevar tablas a base de datos



# Se hizo este ciclo para saber cuántas eran las características ideales para cada modelo

# Crear listas para almacenar resultados
knn_scores = []
tree_scores = []
num_features_list = range(1, len(XKbest.columns) + 1)  # Prueba desde 1 hasta el número máximo de características

# Evaluar K Nearest Neighbors con diferentes números de características
for num_features in num_features_list:
    X_knn = XKbest.iloc[:, :num_features]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn_scores.append(np.mean(cross_val_score(knn, X_knn, yKbest, cv=5)))  # Validación cruzada de 5-fold

# Evaluar Árbol de Decisión con diferentes números de características
for num_features in num_features_list:
    X_tree = XKbest.iloc[:, :num_features]
    tree = DecisionTreeClassifier(random_state=42)
    tree_scores.append(np.mean(cross_val_score(tree, X_tree, yKbest, cv=5)))  # Validación cruzada de 5-fold

# Encontrar el número óptimo de características para cada modelo
best_num_features_knn = num_features_list[np.argmax(knn_scores)]
best_num_features_tree = num_features_list[np.argmax(tree_scores)]

print("Número óptimo de características para K Nearest Neighbors:", best_num_features_knn)
print("Número óptimo de características para Árbol de Decisión:", best_num_features_tree)

# Se descarta poner 60 características en el modelo de árbol de decisión
# porque así no queda mucha interpretabilidad en el caso de estudio
# Se usan 16 características para ambos modelos