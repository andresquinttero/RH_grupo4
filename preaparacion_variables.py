import sqlite3 as sql #### para bases de datos sql
import pandas as pd # para manejo de datos
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import funciones as funciones 

conn = sql.connect("db_empleados") 
df = pd.read_sql("SELECT * FROM all_employees", conn)
df.drop('index', axis = 1, inplace = True)  # delete index column

df['NumCompaniesWorked'] = df['NumCompaniesWorked'].astype(float).astype(int) # Se eliminan los "." como decimales y se convierten a enteros
df['TotalWorkingYears'] = df['TotalWorkingYears'].astype(float).astype(int)

le = LabelEncoder() # Label encoder instance

df['Education'] = le.fit_transform(df['Education'])
df['EnvironmentSatisfaction'] = le.fit_transform(df['EnvironmentSatisfaction'])
df['JobInvolvement'] = le.fit_transform(df['JobInvolvement'])
df['JobLevel'] = le.fit_transform(df['JobLevel'])
df['EducationField'] = le.fit_transform(df['EducationField'])
df['BusinessTravel'] = le.fit_transform(df['BusinessTravel'])
df['Department'] = le.fit_transform(df['Department'])
df['Gender'] = le.fit_transform(df['Gender'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['JobSatisfaction'] = le.fit_transform(df['JobSatisfaction'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
df['Attrition'] = le.fit_transform(df['Attrition'])
df['PerformanceRating'] = le.fit_transform(df['PerformanceRating'])
df['resignationReason'] = le.fit_transform(df['resignationReason'])
df['retirementDate'] = le.fit_transform(df['retirementDate'])
df['retirementType'] = le.fit_transform(df['retirementType'])
df['StockOptionLevel'] = le.fit_transform(df['StockOptionLevel'])
df['WorkLifeBalance'] = le.fit_transform(df['WorkLifeBalance'])

scaler = StandardScaler()

ListaScaler = ['Age','DistanceFromHome', 'MonthlyIncome',
                'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears',
                'TrainingTimesLastYear', 'YearsAtCompany',
                'YearsSinceLastPromotion','YearsWithCurrManager']

for i in ListaScaler:
    scaler = StandardScaler()  # Crea una instancia de StandardScaler para cada columna
    df[[i]] = scaler.fit_transform(df[[i]])

print(df[ListaScaler].describe()) # Comprobando cambios


del df['resignationReason'] # Eliminamos estas variables que ya no nos sirven
del df['retirementDate']
del df['retirementType']

conn = sql.connect("db_empleados") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.

### Llevar tablas a base de datos
df.to_sql("df", conn, if_exists="replace", index=False)
