import sqlite3 as sql #### para bases de datos sql
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import  classification_report
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import a_funciones
conn = sql.connect("db_empleados")

# Conexiones 
df_variables_kbest = pd.read_sql("select * from df_variables_kbest", conn)
df_variables_sfs = pd.read_sql("select * from df_variables_sfs", conn)

# Para las variables seleccionadas con KBest

X = df_variables_kbest.drop("Attrition", axis=1)  # Características
y = df_variables_kbest["Attrition"]  # Variable objetivo

# Se dividen los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

# Modelo K nearest neighbor
knn = KNeighborsClassifier(n_neighbors=3)  # Se crea modelo
knn.fit(X_train, y_train) # Se ajusta modelo
y_predknn = knn.predict(X_test.values) # Predicción conjunto de prueba

# Evaluación
accuracy_knn = accuracy_score(y_test, y_predknn)
report_knn = classification_report(y_test, y_predknn)

print("Accuracy KNN con KBest:", accuracy_knn)
print("Informe de clasificación:\n" ,report_knn)

# Modelo de Árbol de Decisión
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)  # Se ajusta el modelo 

y_pred_tree = tree.predict(X_test)  # Predicción en el conjunto de prueba

# Evaluación del Árbol de Decisión
accuracy_tree = accuracy_score(y_test, y_pred_tree)
report_tree = classification_report(y_test, y_pred_tree)

print("Accuracy del Árbol de Decisión:", accuracy_tree)
print("Informe de clasificación del Árbol de Decisión:\n", report_tree)


# Para las variables seleccionadas con Sequential Feature Selector
X_sfs = df_variables_sfs.drop("Attrition", axis=1)  # Características
y_sfs = df_variables_sfs["Attrition"]  # Variable objetivo

# Se dividen los datos en entrenamiento y prueba
X_train_sfs, X_test_sfs, y_train_sfs, y_test_sfs = train_test_split(X_sfs, y_sfs, test_size=0.2, random_state=42, stratify=y_sfs)

# Modelo K nearest neighbor
knn_sfs = KNeighborsClassifier(n_neighbors=3)  # Se crea modelo 
knn_sfs.fit(X_train_sfs, y_train_sfs)  # Se ajusta modelo 

y_pred_sfs_knn = knn_sfs.predict(X_test_sfs.values)  # Predicción en conjunto de prueba

# Evaluación
accuracy_sfs_knn = accuracy_score(y_test_sfs, y_pred_sfs_knn)
report_sfs_knn = classification_report(y_test_sfs, y_pred_sfs_knn)

print("Accuracy del modelo KNN con SFS:", accuracy_sfs_knn)
print("Informe de clasificación del modelo KNN con SFS:\n", report_sfs_knn)

# Modelo de Árbol de Decisión
tree_sfs = DecisionTreeClassifier(random_state=42)
tree_sfs.fit(X_train_sfs, y_train_sfs)  # Se ajusta el modelo 

y_pred_sfs_tree = tree_sfs.predict(X_test_sfs)  # Predicción en conjunto de prueba

# Evaluación del Árbol de Decisión
accuracy_sfs_tree = accuracy_score(y_test_sfs, y_pred_sfs_tree)
report_sfs_tree = classification_report(y_test_sfs, y_pred_sfs_tree)

print("Accuracy del Árbol de Decisión con SFS:", accuracy_sfs_tree)
print("Informe de clasificación del Árbol de Decisión con SFS:\n", report_sfs_tree)


# Los mejores resultados los da el modelo de Árbol de Decisión 
# con las características de Sequential Feature Selector


# Afinamiento de hiperparámetros

# Define los hiperparámetros a ajustar y sus rangos

param_grid = [{'criterion': ['gini', 'entropy'], # Criterio para medir la calidad de una división
               'max_features': ['auto', 'log2', 'sqrt', 5, 10, 30, 50]},
              {'max_depth': [None, 5, 10, 20, 40, 60, 100],# Profundidad máxima del árbol
               'min_samples_leaf': [1, 2, 3, 4, 5]# Número mínimo de muestras requeridas en un nodo hoja
               }]

grid_search = GridSearchCV(tree_sfs, param_grid, cv=10, scoring='recall') # Busqueda en cuadricula

grid_search.fit(X_train_sfs, y_train_sfs) # Ajusta el modelo

best_params = grid_search.best_params_ #  Mejores parametros
best_tree = grid_search.best_estimator_ #  Mejor modelo

# Evaluación
y_pred_best_tree = best_tree.predict(X_test_sfs)  # Predicción en conjunto de prueba
accuracy_best_tree = accuracy_score(y_test_sfs, y_pred_best_tree)
report_best_tree = classification_report(y_test_sfs, y_pred_best_tree)

print("Mejores hiperparámetros:", best_params)
print("Accuracy del mejor Árbol de Decisión:", accuracy_best_tree)
print("Informe de clasificación del mejor Árbol de Decisión:\n", report_best_tree)

# Evaluación modelo base

# 1. Matriz de Confusión
y_pred = tree_sfs.predict(X_test_sfs)
confusion = confusion_matrix(y_test_sfs, y_pred)
print("Matriz de Confusión:")
print(confusion)

# 2. Precisión
precision = precision_score(y_test_sfs, y_pred)
print("Precisión:", precision)

# 3. Recall
recall = recall_score(y_test_sfs, y_pred)
print("Recall:", recall)

# 4. F1-Score
f1 = f1_score(y_test_sfs, y_pred)
print("F1-Score:", f1)

# 5. Accuracy
accuracy = accuracy_score(y_test_sfs, y_pred)
print("Accuracy:", accuracy)

# 6. Validación Cruzada
scoresbase = cross_val_score(tree_sfs, X_train_sfs, y_train_sfs, cv=10, scoring='accuracy')  # 10-fold cross-validation
print("Resultados de Validación Cruzada (Accuracy):", scoresbase)
print("Precisión media en Validación Cruzada:", scoresbase.mean())


# Evaluación modelo tuneado

# 1. Matriz de Confusión
y_pred_best_tree = best_tree.predict(X_test_sfs)
confusion = a_funciones.show_confusion_matrix(y_test_sfs, y_pred_best_tree)
print("Matriz de Confusión:")
print(confusion)

# 2. Precisión
precision = precision_score(y_test_sfs, y_pred_best_tree)
print("Precisión:", precision)

# 3. Recall
recall = recall_score(y_test_sfs, y_pred_best_tree)
print("Recall:", recall)

# 4. F1-Score
f1 = f1_score(y_test_sfs, y_pred_best_tree)
print("F1-Score:", f1)

# 5. Accuracy
accuracy = accuracy_score(y_test_sfs, y_pred_best_tree)
print("Accuracy:", accuracy)

# 6. Validación Cruzada
scorestuneado = cross_val_score(best_tree, X_train_sfs, y_train_sfs, cv=10, scoring='accuracy')  # 10-fold cross-validation
print("Resultados de Validación Cruzada (Accuracy):", scorestuneado)
print("Precisión media en Validación Cruzada:", scorestuneado.mean())



# Crear un mapa de calor (heatmap) de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="magma", cbar=False)
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.title("Matriz de Confusión")
plt.show()


# El modelo base es muy bueno con los parámetros por defecto,
# el modelo con afinación de hiperparámetros es exactamente el mismo debido a que usamos un random state fijo,
# aún así, se utilizará el modelo tuneado con hiperparámetros.


# Resultados:

# Matriz de Confusión:
# [[731   9]                         9 que el modelo predice que que sí va a salir de la empresa cuando en realidad no saldrá
#  [  2 140]]                        2 que el modelo predice que no va a salir de la empresa cuando en realidad sí saldrá
# Precisión: 0.9395973154362416
# Recall: 0.9859154929577465         el 98.59% de las veces que debería haber predicho la clase positiva, lo hizo correctamente.
# F1-Score: 0.9621993127147765
# Accuracy: 0.9875283446712018
# Resultados de Validación Cruzada (Accuracy): [0.97450425 0.97733711 0.98016997 0.99150142 0.97167139 0.98583569
#  0.97450425 0.98300283 0.98579545 0.98011364]
# Precisión media en Validación Cruzada: 0.9804436003090394


# Guardar el modelo con afinación de hiperparámetros en un archivo .pkl
joblib.dump(best_tree, 'presunto_mejor_arbol.pkl')




