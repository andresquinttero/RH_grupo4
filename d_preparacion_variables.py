import sqlite3 as sql #### para bases de datos sql
import pandas as pd # para manejo de datos
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import a_funciones as a_funciones 

conn = sql.connect("db_empleados") 
df = pd.read_sql("SELECT * FROM all_employees", conn)
df.drop('index', axis = 1, inplace = True)  # delete index column

df['NumCompaniesWorked'] = df['NumCompaniesWorked'].astype(float).astype(int) # Se eliminan los "." como decimales y se convierten a enteros
df['TotalWorkingYears'] = df['TotalWorkingYears'].astype(float).astype(int)

del df['resignationReason'] # Eliminamos estas variables que ya no nos sirven
del df['retirementDate']
del df['retirementType']

# Lista de las columnas para convertir a One-Hot Encoding
columns_to_encode = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'EducationField',
                     'BusinessTravel', 'Department', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                     'PerformanceRating','StockOptionLevel', 'WorkLifeBalance']

# Aplicar One-Hot Encoding a las columnas seleccionadas
df_encoded = pd.get_dummies(df, columns=columns_to_encode)
df = df_encoded.astype(int)

# Aplicar escalado variables numéricas
scaler = StandardScaler()

ListaScaler = ['Age','DistanceFromHome', 'MonthlyIncome',
                'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears',
                'TrainingTimesLastYear', 'YearsAtCompany',
                'YearsSinceLastPromotion','YearsWithCurrManager']

for i in ListaScaler:
    scaler = StandardScaler()  # Crea una instancia de StandardScaler para cada columna
    df[[i]] = scaler.fit_transform(df[[i]])

print(df[ListaScaler].describe()) # Comprobando cambios

conn = sql.connect("db_empleados") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.

### Llevar tablas a base de datos
df.to_sql("df", conn, if_exists="replace", index=False)
