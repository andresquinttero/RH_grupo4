-- Filtrar empleados que se retiraron en 2016
drop table if exists retirement_2016;

create table retirement_2016 as
select
EmployeeID,
retirementDate,
retirementType,
resignationReason
from all_employees
where strftime('%Y', retirementDate) = '2016';

-- Calcular nuevas variables como antigüedad y edad
drop table if exists employee_metrics;

create table employee_metrics as select
EmployeeID,
YearsAtCompany * 365 as TenureDays,
Age,
CASE
WHEN JobLevel < 3 THEN "Junior"
WHEN JobLevel >= 3 AND JobLevel <= 4 THEN "Mid"
ELSE "Senior" END as JobLevelCategory
from all_employees;

-- Crear tabla final para análisis
drop table if exists final_analysis;

create table final_analysis as 
select 
a.*,
b.TenureDays,
b.Age,
b.JobLevelCategory
from retirement_2016 a inner join employee_metrics b on a.EmployeeID = b.EmployeeID;
