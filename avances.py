#### Cargar paquetes siempre al inicio
from platform import python_version ## versión de python
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import sys ## saber ruta de la que carga paquetes

python_version() ### verificar version de python

####################################################################################################################
########################  1. Comprender y limpiar datos ##################################################################
####################################################################################################################

# Cargar el diccionario de datos desde el archivo Excel para entender los campos de las bases de datos
data_dictionary_path = '/data_dictionary.xlsx'
data_dictionary_df = pd.read_excel(data_dictionary_path)

# Mostrar el diccionario de datos
data_dictionary_df.head(len(data_dictionary_df))

########   Verificar lectura correcta de los datos
########   Verificar Datos faltantes (eliminar variables si es necesario) (la imputación está la parte de modelado)
########   Tipos de variables (categoricas/numéricas/fechas)
########   Niveles en categorícas 
########   Observaciones por categoría
########   Datos atípicos en numéricas


### Cargar tablas de datos desde github ###
#action=("data//tbl_Action.csv")  
#employees=("data//tbl_Employee.csv")  
#performance=("data//tbl_Perf.csv")   
