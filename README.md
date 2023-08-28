# **Proyecto de Análisis de Rotación de Empleados**

## **Grupo de Trabajo**
- **Juan Diego Correa**
- **Santiago Pianda**
- **Andrés Quintero**

---

## **Descripción del Proyecto**
Este proyecto tiene como objetivo analizar la alta tasa de rotación de empleados en una empresa con aproximadamente 4000 empleados. Utilizamos diferentes técnicas de análisis de datos y machine learning para identificar las causas y proponer soluciones.

---

## **Estructura de Archivos**

### **Código Fuente**

#### `funciones_updated.py`
Este archivo contiene **funciones de utilidad general** utilizadas en todo el proyecto. Incluye una función para ejecutar archivos SQL, así como funciones para la normalización de datos.

#### `preprocesamiento.py`
Este archivo contiene el código para el **preprocesamiento de los datos**. Se encarga de la limpieza y transformación de los datos para su posterior análisis.

---

### **Datos**

#### `employee_survey_data.csv`
Contiene los resultados de una **encuesta realizada a los empleados** sobre su satisfacción laboral.

#### `general_data.csv`
Incluye **información general sobre los empleados**, como edad, departamento, nivel de educación, entre otros.

#### `manager_survey_data.csv`
Contiene los resultados de una **encuesta de desempeño realizada por los jefes** de los empleados.

#### `retirement_info.csv`
Incluye información sobre los **empleados que se han retirado** de la empresa, con detalles como la fecha de retiro y el motivo.

#### `db_empleados`
Base de datos SQLite que contiene todas las **tablas de datos necesarias para el proyecto**.

---

### **Consultas SQL**

#### `preprocesamientos_updated.sql`
Archivo SQL que contiene **varias consultas para el preprocesamiento de los datos**. Se encarga de crear nuevas tablas y variables que serán utilizadas en el análisis.

---

## **Cómo Empezar**
1. **Asegúrate de tener todas las dependencias instaladas**.
2. **Ejecuta `preprocesamiento.py`** para limpiar y transformar los datos.
3. **Utiliza `funciones_updated.py`** según sea necesario para tareas adicionales como ejecución de consultas SQL.
