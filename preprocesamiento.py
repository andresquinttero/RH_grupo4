import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import funciones as funciones  ###archivo de funciones propias

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
df_general.sort_values(by=['EmployeeID'],ascending=1).head(5)
df_manager.sort_values(by=['EmployeeID'],ascending=1).head(5)
df_retirement.sort_values(by=['EmployeeID'],ascending=1).head(5)

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
print(df.columns)

##### Aqui se solucionan los nulos de algunas columnas #####

# Se cambia a cero los valores nulos de la columna 'NumCompaniesWorked' asumiendo que esta es su primera empresa y habiendo verificado que todos iniciaron siendo mayores de edad
df[df['NumCompaniesWorked'].isnull()]
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(0)
# Para la columna 'EnvironmentSatisfaction' se observó la mediana y el promedio y se llegó a la conclusión de que 3 es el valor más adecuado para no interferir de una
# manera significativa en el resultado y por lo tanto los valores nulos se remplazarán por 3
df['EnvironmentSatisfaction'].value_counts()
df['EnvironmentSatisfaction'].mean()
df['EnvironmentSatisfaction'].median()
df[df['EnvironmentSatisfaction'].isnull()]
df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].fillna(3)
# Para la columna 'JobSatisfaction' se observo el promedio y un conteo de repeticion y se llego a la conclusion de que el 3 era el valor más adecuado para no interferir de una
# manera significativa en el resultado y por lo tanto los valores nulos se remplazaran por 3
df['JobSatisfaction'].value_counts()
df['JobSatisfaction'].mean()
df['JobSatisfaction'].median()
df[df['JobSatisfaction'].isnull()]
df['JobSatisfaction'] = df['JobSatisfaction'].fillna(3)
# Para la columna 'WorkLifeBalance' se observo el promedio y un conteo de repeticion y se llego a la conclusion de que el 3 era el valor maas adecuado para no interferir de una
# manera significativa en el resultado y por lo tanto los valores nulos se remplazaran por 3
df['WorkLifeBalance'].value_counts()
df['WorkLifeBalance'].mean()
df['WorkLifeBalance'].median()
df[df['WorkLifeBalance'].isnull()]
df['WorkLifeBalance'] = df['WorkLifeBalance'].fillna(3)
# Remplazar los valores de los años trabajados en la empresa en la columna de años totales trabajdos para quitar los nulos
df['TotalWorkingYears'].isnull().sum()
df['TotalWorkingYears'].fillna(df['YearsAtCompany'], inplace=True)

#Al volver a revisar vemos que las columnas retirementDate, retirementType, y resignationReason tienen valores nulos, lo cual es esperado dado que no todos los empleados se habrán retirado o renunciado.
print(df.isnull().sum())

#La columna retirementDate contiene fechas en formato de cadena de texto (string), que deberían convertirse a un formato de fecha.
df['retirementDate'].info()
df['retirementDate'] = pd.to_datetime(df['retirementDate'], dayfirst=True)

#Las columnas retirementType y resignationReason contienen valores categóricos en formato de cadena de texto.
print(df['retirementType'].dtype)
print(df['resignationReason'].dtype)
df['retirementType'] = df['retirementType'].astype('category')
df['resignationReason'] = df['resignationReason'].astype('category')

#Ahora ya tienen un formato adecuado
print(df.dtypes)

# Ahora se eliminaran las columnas que no aportan al analisis
df.drop(columns=["EmployeeCount", "Over18", "StandardHours"],inplace=True)


##### SQL #####
conn = sql.connect("db_empleados")  # creacion de la base de datos
cursor = conn.cursor() # para funcion execute
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
cursor.fetchall() # para ver en tablas las bases de datos

# Llevar DataFrame a la base de datos
df.to_sql("all_employees", conn, if_exists="replace")

## Hacer consultas para verificar la base de datos ##

#Numero de empleados por departamento
pd.read_sql("""SELECT Department, COUNT(*) 
                                FROM all_employees 
                                GROUP BY Department""", conn)

#Retiros segun departamento
pd.read_sql("""SELECT Department, COUNT(*) as Retirements 
                                    FROM all_employees 
                                    WHERE retirementDate IS NOT NULL 
                                    GROUP BY Department""", conn)

#Retiros según el nivel de satisfacción laboral
pd.read_sql("""SELECT JobSatisfaction, COUNT(*) as Retirements 
                                    FROM all_employees 
                                    WHERE retirementDate IS NOT NULL 
                                    GROUP BY JobSatisfaction""", conn)

#Retiros según la antigüedad en la empresa
pd.read_sql("""SELECT YearsAtCompany, COUNT(*) as Retirements 
                                    FROM all_employees 
                                    WHERE retirementDate IS NOT NULL 
                                    GROUP BY YearsAtCompany
                                    ORDER BY YearsAtCompany""", conn)

#Retiros según la edad
pd.read_sql("""SELECT Age, COUNT(*) as Retirements 
                                    FROM all_employees 
                                    WHERE retirementDate IS NOT NULL 
                                    GROUP BY Age
                                    ORDER BY Retirements DESC""", conn)

#Razones de retiro
pd.read_sql("""SELECT resignationReason, COUNT(*) as Count 
                                    FROM all_employees 
                                    WHERE retirementDate IS NOT NULL 
                                    GROUP BY resignationReason""", conn)

#Analizar los retiros por fecha exacta
pd.read_sql("""SELECT retirementDate AS Date, 
                        COUNT(*) AS Retirements 
                        FROM all_employees 
                        WHERE retirementDate IS NOT NULL 
                        GROUP BY Date
                        ORDER BY Retirements DESC""", conn)

# Rellenar los valores NaN con "No" para la columna Attrition
df['Attrition'].fillna('No', inplace=True)

# Mapear "Yes" a 1 y "No" a 0
attrition_mapping = {'Yes': 1, 'No': 0}
df['Attrition'] = df['Attrition'].map(attrition_mapping)

# Verificar si la conversión fue exitosa
df['Attrition'].unique()

#### Las variables identificadas para recategorizar son: 
# Pendiente, Age puede ser una...

# Llamar a la función para identificar y eliminar outliers
funciones.identify_and_remove_outliers(conn, ['MonthlyIncome', 'TrainingTimesLastYear', 'YearsAtCompany', 'TotalWorkingYears'])

# Lista de columnas para cambiar el tipo de datos
columns_to_convert = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobSatisfaction', 'PerformanceRating', 'WorkLifeBalance', 'JobLevel'
]

# Cambiar el tipo de datos de cada columna a 'str'
for column in columns_to_convert:
    df[column] = df[column].astype(str)

###### Preprocesamientos que se realizarán con SQL:

##### 1. Filtrar empleados que se retiraron en 2016 para analizar las razones detrás de su retiro.
##### 2. Calcular nuevas variables como  la antigüedad de los empleados en la empresa, la edad de los empleados, y otras métricas como la satisfacción media a lo largo del tiempo.

#Para hacer todos los preprocesamienteos se crea archivo .sql que se ejecuta con la función: ejecutar_sql del archivo funciones.py
df.info()
df.describe(include='all')

df.to_sql("all_employees", conn, if_exists="replace")

df = pd.read_sql("SELECT * FROM all_employees", conn)
cur=conn.cursor()
funciones.ejecutar_sql('preprocesamientos.sql',cur)
