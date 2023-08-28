import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql

####################################################################################################################
########################  1. Comprender y limpiar datos ##################################################################
####################################################################################################################

# Cargar las bases de datos en DataFrames
df_employee= pd.read_csv('databases/employee_survey_data.csv')
df_manager = pd.read_csv('databases/manager_survey_data.csv')
# A estos 2 dataframes se cambio el delimitador ';' por ','
df_general = pd.read_csv('databases/general_data.csv')
df_retirement = pd.read_csv('databases/retirement_info.csv')

# Cargar el diccionario de datos desde el archivo Excel para entender los campos de las bases de datos
df_data_dictionary = pd.read_excel('databases/data_dictionary.xlsx')
# Mostrar el diccionario de datos
df_data_dictionary.head(len(df_data_dictionary))

###### Verificar lectura correcta de los datos
df_employee.sort_values(by=['EmployeeID'],ascending=1).head(5)
df_general.sort_values(by=['EmployeeID'],ascending=0).head(5)
df_manager.sort_values(by=['EmployeeID'],ascending=0).head(5)
df_retirement.sort_values(by=['EmployeeID'],ascending=0).head(5)

conn = sql.connect(':memory:')

# Insertar DataFrames en la base de datos SQLite
employee_survey_data.to_sql('employee_survey_data', conn, index=False)
general_data.to_sql('general_data', conn, index=False)
in_time.to_sql('in_time', conn, index=False)
manager_survey_data.to_sql('manager_survey_data', conn, index=False)
out_time.to_sql('out_time', conn, index=False)
retirement_info.to_sql('retirement_info', conn, index=False)

# Cerrar la conexión
conn.close()
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
