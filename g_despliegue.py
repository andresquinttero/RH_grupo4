import joblib

# Cargar el modelo desde el archivo .pkl
tree = joblib.load("presunto_mejor_arbol.pkl")

# En el conjunto de datos 'X_datos2024' se podrán hacer predicciones en el futuro,
# cuando se tengan más datos en recursos humanos

#          predictions = tree.predict(X_datos2024)

# Luego "predictions" contendrá las predicciones del modelo para los nuevos datos
