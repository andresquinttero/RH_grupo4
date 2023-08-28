-- query1.sql

SELECT * 
FROM general_data AS g
JOIN employee_survey_data AS e ON g.EmployeeID = e.EmployeeID
JOIN manager_survey_data AS m ON g.EmployeeID = m.EmployeeID;

-- query2.sql

SELECT 
    COUNT(CASE WHEN columna1 IS NULL THEN 1 END) AS columna1_nulls,
    COUNT(CASE WHEN columna2 IS NULL THEN 1 END) AS columna2_nulls
FROM (
    SELECT * 
    FROM general_data AS g
    JOIN employee_survey_data AS e ON g.EmployeeID = e.EmployeeID
    JOIN manager_survey_data AS m ON g.EmployeeID = m.EmployeeID
) AS combined_data;
