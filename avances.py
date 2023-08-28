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

# Obtener información sobre los DataFrames
df_employee.info(verbose=True)
df_general.info()
df_manager.info()
df_retirement.info()

# Verificar entradas faltantes
print(df_employee.isnull().sum())
print(df_general.isnull().sum())
print(df_manager.isnull().sum())
print(df_retirement.isnull().sum())

##### Unir los diferentes dataframes utilizando la columna 'EmployeeID' como clave
df = df_employee.merge(df_general, on='EmployeeID', how='inner')\
                        .merge(df_manager, on='EmployeeID', how='inner')\
                        .merge(df_retirement, on='EmployeeID', how='left')

# Procedemos a revisar los valores nulos
print(df.isnull().sum())
df[df['TotalWorkingYears'].isnull()] 

##### Aqui se solucionan los nulos de algunas columnas #####

# Se cambia a cero los valores nulos de la columna 'NumCompaniesWorked' asumiendo que esta es su primera empresa y habiendo verificado que todos iniciaron siendo mayores de edad
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(0)
# Para la columna 'EnvironmentSatisfaction' se observó la mediana y el promedio y se llegó a la conclusión de que 3 es el valor más adecuado para no interferir de una
# manera significativa en el resultado y por lo tanto los valores nulos se remplazarán por 3
df['EnvironmentSatisfaction'].value_counts()
df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].fillna(3)
# Para la columna 'JobSatisfaction' se observo el promedio y un conteo de repeticion y se llego a la conclusion de que el 4 era el valor más adecuado para no interferir de una
# manera significativa en el resultado y por lo tanto los valores nulos se remplazaran por 4
df['JobSatisfaction'].value_counts()
df['JobSatisfaction'] = df['JobSatisfaction'].fillna(4)
# Para la columna 'WorkLifeBalance' se observo el promedio y un conteo de repeticion y se llego a la conclusion de que el 3 era el valor maas adecuado para no interferir de una
# manera significativa en el resultado y por lo tanto los valores nulos se remplazaran por 3
df['WorkLifeBalance'].value_counts()
df['WorkLifeBalance'] = df['WorkLifeBalance'].fillna(3)
# Remplazar los valores de los años trabajados en la empresa en la columan de años totales trabajdos para quitar los nulos
df['TotalWorkingYears'].fillna(df['YearsAtCompany'], inplace=True)

#Al volver a revisar vemos que las columnas retirementDate, retirementType, y resignationReason tienen valores nulos, lo cual es esperado dado que no todos los empleados se habrán retirado o renunciado.
print(df.isnull().sum())

#La columna retirementDate contiene fechas en formato de cadena de texto (string), que deberían convertirse a un formato de fecha.
df['retirementDate'] = pd.to_datetime(df['retirementDate'], dayfirst=True)

#Las columnas retirementType y resignationReason contienen valores categóricos en formato de cadena de texto.
df['retirementType'] = df['retirementType'].astype('category')
df['resignationReason'] = df['resignationReason'].astype('category')

#Ahora ya tienen un formato adecuado
print(df.dtypes)

# Ahora se eliminaran las columnas que no aportan al analisis
df.drop(columns=["EmployeeCount", "Over18", "StandardHours","EmployeeID"],inplace=True)




