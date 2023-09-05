#### Cargar paquetes siempre al inicio
import pandas as pd  # para manejo de datos
import sqlite3 as sql  # para bases de datos SQL
import a_funciones as a_funciones  # archivo de funciones propias
import matplotlib.pyplot as plt  # gráficos
import seaborn as sns  # gráficos adicionales
from pandas.plotting import scatter_matrix  # para matriz de correlaciones
from sklearn import tree  # para ajustar árboles de decisión
from sklearn.tree import export_text  # para exportar reglas del árbol
from sklearn.tree import DecisionTreeClassifier

# Conexión a la base de datos
conn = sql.connect("db_empleados")
cur = conn.cursor()  # para ejecutar consultas SQL en la base de datos


# Cargar datos desde SQL
df = pd.read_sql("select * from all_employees", conn)

### Explorar variable respuesta ###
# Visualización de la distribución de la edad
sns.histplot(df['Age'], kde=False)
plt.title('Distribución de edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Visualización de la distribución de la satisfacción laboral
sns.histplot(df['JobSatisfaction'], kde=False)
plt.title('Distribución de satisfacción laboral')
plt.xlabel('Satisfacción laboral')
plt.ylabel('Frecuencia')
plt.show()

# Convertir 'JobSatisfaction' a tipo int
df['JobSatisfaction'].dtype
df['JobSatisfaction'] = df['JobSatisfaction'].astype(float).astype(int)
df['JobSatisfaction'].unique()

# Visualización de la relación entre la satisfacción laboral y la retención del empleado
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title('Relación entre satisfacción laboral y retención del empleado')
plt.xlabel('Abandono')
plt.ylabel('Satisfacción laboral')
plt.show()

# Análisis exploratorio para obtener una idea de las características de los empleados que abandonan la empresa y aquellos que permanecen en ella.
plt.figure(figsize=(15, 6))
sns.countplot(x="Age", hue="Attrition", data=df)
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.show()

# Ver la relación entre la variable objetivo y la edad
plt.figure(figsize=(8, 6))
sns.boxplot(x='Attrition', y='Age', data=df)
plt.title('Relación entre la variable objetivo y la edad')
plt.xlabel('Abandono')
plt.ylabel('Edad')
plt.show()

# Distribución de retiro por género y departamento
plt.figure(figsize=(10, 6))
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Distribución de retiro por Departamento")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x="Gender", hue="Attrition", data=df)
plt.title("Distribución de retiro por Género")
plt.show()

# Satisfacción laboral por departamento
plt.figure(figsize=(10, 6))
sns.boxplot(x="Department", y="JobSatisfaction", data=df)
plt.title("Satisfacción laboral por Departamento")
plt.show()

# Edad vs. Años en la compañía
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Age", y="YearsAtCompany", hue="Attrition", data=df)
plt.title("Edad vs. Años en la compañía")
plt.show()

# Boxplot de años con el gerente actual según la retención
plt.figure(figsize=(10, 6))
sns.boxplot(x="Attrition", y="YearsWithCurrManager", data=df)
plt.title("Años con el gerente actual según la retención")
plt.show()


# Mantener solo columnas numéricas
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Correlación de variables
corr_matrix = df_numeric.corr()
attrition_correlation = corr_matrix["Attrition"].sort_values(ascending=False)
attrition_correlation

##### ANALISIS #######
#Correlaciones Positivas:
# NumCompaniesWorked (0.042) y PercentSalaryHike (0.033) tienen correlaciones positivas muy bajas con Attrition.
# Esto sugiere que los empleados que han trabajado en más empresas o que han tenido 
# aumentos de salario más significativos son un poco más propensos a abandonar la empresa. 
# Sin embargo son correalaciones muy débiles que sugieren que no hay una relación significativa con Attrition

#Correlaciones Negativas:
# TotalWorkingYears (-0.171), Age (-0.159), y YearsWithCurrManager (-0.156) tienen correlaciones negativas moderadas. 
# Esto podría indicar que los empleados más antiguos, de mayor edad, o con más tiempo con su gerente actual
# son menos propensos a abandonar la empresa.
# Variables como MonthlyIncome (-0.031), YearsSinceLastPromotion (-0.033), y TrainingTimesLastYear (-0.049) 
# también muestran una correlación negativa, aunque más baja.
#Correlaciones Cercanas a Cero:

# Variables como EmployeeID, StockOptionLevel, DistanceFromHome, y JobLevel tienen correlaciones muy cercanas
# a cero, lo que sugiere que probablemente no tengan un impacto significativo en la rotación de empleados.


# Eliminamos las variables no numéricas y los NA para el modelo
df_tree = df.select_dtypes(include=['float64', 'int64']).dropna()

#Evaluar la importancia de las diferentes variables numéricas en la predicción de la variable "Attrition" utilizando un modelo de árbol de decisión.
X = df_tree.drop("Attrition", axis=1)  # Asumiendo que 'Attrition' está codificado como 0 y 1
y = df_tree["Attrition"]

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

print("Importancia de las variables según el árbol de decisión:\n", feature_importance_df.sort_values(by="Importance", ascending=False))


### Analisis adicionales para attrition

# Visualización de la distribución de "TotalWorkingYears"
sns.histplot(df['TotalWorkingYears'], kde=False)
plt.title('Distribución de TotalWorkingYears')
plt.xlabel('TotalWorkingYears')
plt.ylabel('Frecuencia')
plt.show()

# Visualización de la distribución de "Age"
sns.histplot(df['Age'], kde=False)
plt.title('Distribución de Age')
plt.xlabel('Age')
plt.ylabel('Frecuencia')
plt.show()

# Visualización de la distribución de "YearsWithCurrManager"
sns.histplot(df['YearsWithCurrManager'], kde=False)
plt.title('Distribución de YearsWithCurrManager')
plt.xlabel('YearsWithCurrManager')
plt.ylabel('Frecuencia')
plt.show()

#------------------------------------------------------------------------

# Según el arbol de decisión, estas son las 4 variables más importantes respecto a la objetivo, 
# Con estos gráficos podemos ver la relación directa con el abandono de los empleados de la empresa.

# Relación entre la variable objetivo Attrition y TotalWorkingYears
sns.boxplot(x='Attrition', y='TotalWorkingYears', data=df)
plt.title('Relación entre la variable objetivo y TotalWorkingYears')
plt.xlabel('Abandono')
plt.ylabel('Total de años trabajados')
plt.show()

# Relación entre "Age" y "Attrition"
sns.boxplot(x='Attrition', y='Age', data=df)
plt.title('Relación entre Age y Attrition')
plt.xlabel('Attrition')
plt.ylabel('Age')
plt.show()

# Relación entre "NumCompaniesWorked" y "Attrition"
sns.boxplot(x='Attrition', y='NumCompaniesWorked', data=df)
plt.title('Relación entre número de empresas trabajadas y Attrition')
plt.xlabel('Attrition')
plt.ylabel('NumCompaniesWorked')
plt.show()

# Relación entre "MonthlyIncome" y "Attrition"
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title('Relación entre el salario mensual y Attrition')
plt.xlabel('Attrition')
plt.ylabel('MonthlyIncome')
plt.show()

#------------------------------------------------------------------------

# Relación entre "YearsWithCurrManager" y "Attrition"
sns.boxplot(x='Attrition', y='YearsWithCurrManager', data=df)
plt.title('Relación entre YearsWithCurrManager y Attrition')
plt.xlabel('Attrition')
plt.ylabel('YearsWithCurrManager')
plt.show()

# Relación entre "JobSatisfaction" y "Attrition"
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title('Relación entre JobSatisfaction y Attrition')
plt.xlabel('Attrition')
plt.ylabel('JobSatisfaction')
plt.show()

# Relación entre "EnvironmentSatisfaction" y "Attrition"
ax = sns.countplot(data=df, x='EnvironmentSatisfaction', hue='Attrition', palette=["dodgerblue", "firebrick"])

plt.xlabel('EnvironmentSatisfaction')
plt.ylabel('Frecuencia')
plt.title('Gráfico de Barras: EnvironmentSatisfaction vs. Attrition')
plt.legend(title='Attrition', labels=['No', 'Sí'])

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

plt.show()


##### Análisis final

# Mayor edad y más años de trabajo disminuyen la probabilidad de abandono: Las variables Age y TotalWorkingYears 
# tienen una correlación negativa moderada con Attrition, también fueron las variables más importantes según 
# nuestro arbol de decisión, lo cuál sugiere que los empleados más jóvenes y con menos experiencia son más
# propensos a abandonar la empresa.

# La gestión y el tiempo en la empresa son clave: YearsWithCurrManager tiene una correlación negativa de -0.156 
# con Attrition. Esto puede implicar que una buena relación con el gerente actual o simplemente el tiempo 
# pasado con el mismo gerente puede llevar a una menor tasa de rotación.

# Satisfacción laboral y ambiental son importantes: Las variables JobSatisfaction y EnvironmentSatisfaction 
# tienen correlaciones negativas de -0.104 y -0.101, respectivamente. Esto indica que cuanto mayor es la 
# satisfacción laboral y ambiental, menor es la probabilidad de que un empleado abandone la empresa.

# Cambios frecuentes de empresa podrían ser una señal de alerta: NumCompaniesWorked tiene una correlación 
# positiva de 0.041 con Attrition. Aunque no es una correlación fuerte, podría ser un indicador de que aquellos 
# empleados que han cambiado de empresa con más frecuencia en el pasado son más propensos a abandonar.

# Aumento salarial no garantiza retención: Sorprendentemente, PercentSalaryHike tiene una correlación 
# positiva baja (0.032) con Attrition, lo que sugiere que un aumento salarial no necesariamente garantiza
# la retención del empleado.

# Nivel educativo y nivel laboral no son factores críticos: Las variables como Education y JobLevel tienen 
# correlaciones muy cercanas a cero con Attrition, lo que sugiere que no son factores críticos en la
# decisión de un empleado de abandonar o quedarse en la empresa.
