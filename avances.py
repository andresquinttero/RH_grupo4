import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import sys ## saber ruta de la que carga paquetes

####################################################################################################################
########################  1. Comprender y limpiar datos ##################################################################
####################################################################################################################

# Cargar el diccionario de datos desde el archivo Excel para entender los campos de las bases de datos
data_dictionary_path = 'databases/data_dictionary.xlsx'
data_dictionary_df = pd.read_excel(data_dictionary_path)

# Mostrar el diccionario de datos
data_dictionary_df.head(len(data_dictionary_df))

# Rutas de los archivos CSV
employee_survey_data_path = 'databases/employee_survey_data.csv'
general_data_path = 'databases/general_data.csv'
in_time_path = 'databases/in_time.csv'
manager_survey_data_path = 'databases/manager_survey_data.csv'
out_time_path = 'databases/out_time.csv'
retirement_info_path = 'databases/retirement_info.csv'

# Cargar los archivos CSV en DataFrames de Python
employee_survey_data_df = pd.read_csv(employee_survey_data_path)
general_data_df = pd.read_csv(general_data_path)
in_time_df = pd.read_csv(in_time_path)
manager_survey_data_df = pd.read_csv(manager_survey_data_path)
out_time_df = pd.read_csv(out_time_path)
retirement_info_df = pd.read_csv(retirement_info_path)


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
