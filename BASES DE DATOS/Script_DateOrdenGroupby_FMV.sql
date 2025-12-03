-- SCRIPT DE CONSULTAS SQL - SISTEMA DE BIBLIOTECA
-- 1. CONSULTAS CON FUNCIONES DE FECHA
-- 1.1 Que fecha sera dentro de una semana?
SELECT CURRENT_DATE + INTERVAL '7 days' AS fecha_en_una_semana;

-- 1.2 Mostrar "el dia de hoy [fecha] es [dia_semana]"
SELECT 'El dia de hoy ' || TO_CHAR(CURRENT_DATE, 'DD/MM/YYYY') || ' es ' || 
       TO_CHAR(CURRENT_DATE, 'Day') AS mensaje_fecha;

-- Consultar todos los prestamos realizados en los ultimos 7 dias
-- no habia registros que coincidieran entonces se poblará
INSERT INTO prestamos(id_usuario, id_libro, fecha_prestamo, fecha_devolucion_max, estado_prestamo)
VALUES 
(8, 15, '2025-06-15', '2025-06-30', 'pendiente'); -- Retraso, no puede recibir nuevos préstamos
SELECT * FROM prestamos 
WHERE fecha_prestamo >= CURRENT_DATE - INTERVAL '7 days';

-- Obtener todos los usuarios cuyo registro ocurrio en los ultimos 6 meses
SELECT id_usuario, nombre, fecha_registro FROM usuarios 
WHERE fecha_registro >= CURRENT_DATE - INTERVAL '6 months';

-- Listar los libros cuyo anio de publicacion sea hace mas de 10 anios
SELECT * FROM libros 
WHERE anio_publicacion < EXTRACT(YEAR FROM CURRENT_DATE) - 10;

-- Seleccionar todos los prestamos con retraso que debian devolverse hace mas de 15 dias
SELECT * FROM prestamos 
WHERE estado_prestamo = 'con retraso' 
AND fecha_devolucion_max < CURRENT_DATE - INTERVAL '15 days';

-- Obtener los usuarios que cumplan anios este mes
SELECT * FROM usuarios 
WHERE EXTRACT(MONTH FROM fecha_nacimiento) = EXTRACT(MONTH FROM CURRENT_DATE);

-- Mostrar los prestamos realizados en el primer trimestre del anio
-- no habia entonces insertamos algunos
INSERT INTO prestamos(id_usuario, id_libro, fecha_prestamo, fecha_devolucion_max, estado_prestamo)
VALUES 
(7, 22, '2025-01-15', '2025-01-30', 'pendiente'),
(8, 17, '2025-02-15', '2025-03-01', 'devuelto'); 
SELECT * FROM prestamos 
WHERE EXTRACT(MONTH FROM fecha_prestamo) BETWEEN 1 AND 3
AND EXTRACT(YEAR FROM fecha_prestamo) = EXTRACT(YEAR FROM CURRENT_DATE);

-- Listar los usuarios registrados en diciembre de cualquier anio
-- no habia entonces inserto algunos
INSERT INTO usuarios (nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento,fecha_registro, id_rol, contrasena)
VALUES
('Juanita', 'Soriano', 'Salazar', 'juanita.soriano@example.com', '5551234509', '2001-08-13','2015-12-02', 3, 'PassJuanita123');
SELECT id_usuario,nombre,fecha_registro FROM usuarios 
WHERE EXTRACT(MONTH FROM fecha_registro) = 12;

-- Mostrar los prestamos cuya fecha de prestamo fue en un fin de semana
SELECT * FROM prestamos 
WHERE EXTRACT(DOW FROM fecha_prestamo) IN (0, 6); -- 0=Domingo, 6=Sabado

-- Listar libros publicados en la primera mitad del anio (NO SE PUEDE PORQUE SOLO ALMACENA AÑO)

-- Seleccionar todos los usuarios que nacieron en los anios bisiestos
SELECT nombre, fecha_nacimiento FROM usuarios 
WHERE (EXTRACT(YEAR FROM fecha_nacimiento) % 4 = 0 AND EXTRACT(YEAR FROM fecha_nacimiento) % 100 != 0) 
   OR (EXTRACT(YEAR FROM fecha_nacimiento) % 400 = 0);

-- Consultar prestamos realizados en el ultimo trimestre del anio pasado
-- no habia entonces insertamos algunos
INSERT INTO prestamos(id_usuario, id_libro, fecha_prestamo, fecha_devolucion_max, estado_prestamo)
VALUES 
(11, 17, '2024-10-15', '2024-10-30', 'pendiente'),
(24, 15, '2024-11-15', '2025-11-30', 'devuelto'); 
SELECT * FROM prestamos 
WHERE EXTRACT(MONTH FROM fecha_prestamo) BETWEEN 10 AND 12
AND EXTRACT(YEAR FROM fecha_prestamo) = EXTRACT(YEAR FROM CURRENT_DATE) - 1;

-- Obtener libros publicados hace exactamente 15 anios
--no habia entonces insertar
INSERT INTO libros (isbn, titulo, anio_publicacion, editorial, edicion, cantidad_total, cantidad_disponible, clasificacion)
VALUES
('978-8-40-814830-2', '#Hiperconectados', 2010, 'Sudamericana', 1, 10, 7, 5);
SELECT * FROM libros 
WHERE anio_publicacion = EXTRACT(YEAR FROM CURRENT_DATE) - 15;

-- Listar los usuarios cuyo cumpleanios sea en los proximos 30 dias
SELECT * FROM usuarios 
WHERE DATE_PART('doy', fecha_nacimiento) BETWEEN DATE_PART('doy', CURRENT_DATE) 
AND DATE_PART('doy', CURRENT_DATE + INTERVAL '30 days');

-- Seleccionar los prestamos cuya fecha de prestamo fue hace mas de 6 meses pero menos de 1 anio
SELECT * FROM prestamos 
WHERE fecha_prestamo < CURRENT_DATE - INTERVAL '6 months'
AND fecha_prestamo > CURRENT_DATE - INTERVAL '1 year';

-- 2. ORDENAMIENTO Y LIMITE DE RESULTADOS

-- Mostrar a las personas ordenadas alfabeticamente por nombre
SELECT * FROM usuarios 
ORDER BY nombre ASC;

-- Mostrar las primeras 3 personas que son lectores
SELECT u.* FROM usuarios u
JOIN roles r ON u.id_rol = r.id_rol
WHERE r.nombre_rol = 'lector'
ORDER BY u.nombre
LIMIT 3;

-- Obtener los 10 usuarios mas recientes registrados en el sistema
SELECT nombre, fecha_registro FROM usuarios 
ORDER BY fecha_registro DESC 
LIMIT 10;

-- Listar los 5 libros mas prestados en el sistema
SELECT l.*, COUNT(p.id_prestamo) AS total_prestamos
FROM libros l
LEFT JOIN prestamos p ON l.id_libro = p.id_libro
GROUP BY l.id_libro
ORDER BY total_prestamos DESC
LIMIT 5;

-- Listar los 10 libros publicados mas recientemente
SELECT * FROM libros 
WHERE anio_publicacion IS NOT NULL
ORDER BY anio_publicacion DESC 
LIMIT 10;

-- Mostrar las primeras 15 multas generadas en el sistema, ordenadas por fecha
INSERT INTO multas(id_prestamo, monto, pagado)
VALUES (3, 10.99, FALSE);
SELECT * FROM multas 
ORDER BY fecha_multa ASC 
LIMIT 15;

-- Obtener los 5 autores con mas libros registrados en el sistema
SELECT a.*, COUNT(la.id_libro) AS total_libros
FROM autores a
LEFT JOIN libroAutor la ON a.id_autor = la.id_autor
GROUP BY a.id_autor
ORDER BY total_libros DESC
LIMIT 5;

-- Listar los 8 libros mas antiguos disponibles en la biblioteca
SELECT * FROM libros 
WHERE anio_publicacion IS NOT NULL AND cantidad_disponible > 0
ORDER BY anio_publicacion ASC 
LIMIT 8;

-- Obtener los primeros 10 prestamos mas recientes en el sistema
SELECT * FROM prestamos 
ORDER BY fecha_prestamo DESC 
LIMIT 10;

-- Mostrar los 5 usuarios con las contrasenias mas largas, ordenados de mayor a menor longitud
SELECT id_usuario,nombre, primer_apellido,segundo_apellido,contrasena, LENGTH(contrasena) AS longitud_contrasena
FROM usuarios 
ORDER BY LENGTH(contrasena) DESC 
LIMIT 5;

-- Obtener las primeras 10 editoriales con mas libros publicados, ordenadas alfabeticamente
SELECT editorial, COUNT(*) AS total_libros
FROM libros 
WHERE editorial IS NOT NULL
GROUP BY editorial
ORDER BY total_libros DESC, editorial ASC
LIMIT 10;

-- Listar los 12 primeros libros clasificados con la puntuacion mas alta
SELECT * FROM libros 
ORDER BY clasificacion DESC 
LIMIT 12;

-- Mostrar las 15 primeras devoluciones que se hicieron mas rapidamente despues del prestamo
--no habia entonces insertar
INSERT INTO devoluciones(id_prestamo, fecha_devolucion) VALUES
(1,'2024-01-23'),
(4,'2024-01-19'),
(5,'2024-05-22');

SELECT d.*, p.fecha_prestamo, 
       (d.fecha_devolucion - p.fecha_prestamo) AS dias_prestamo
FROM devoluciones d
JOIN prestamos p ON d.id_prestamo = p.id_prestamo
ORDER BY (d.fecha_devolucion - p.fecha_prestamo) ASC
LIMIT 15;

-- Listar los 5 libros mas prestados en los ultimos 6 meses, ordenados por la cantidad de prestamos
SELECT l.*, COUNT(p.id_prestamo) AS prestamos_recientes
FROM libros l
LEFT JOIN prestamos p ON l.id_libro = p.id_libro
WHERE p.fecha_prestamo >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY l.id_libro
ORDER BY prestamos_recientes DESC
LIMIT 5;

-- 3. AGRUPAMIENTO

-- Cuantas personas estan registradas de cada rol tenemos?
SELECT r.nombre_rol, COUNT(u.id_usuario) AS total_usuarios
FROM roles r
LEFT JOIN usuarios u ON r.id_rol = u.id_rol
GROUP BY r.id_rol, r.nombre_rol
ORDER BY total_usuarios DESC;

-- Obtener la cantidad de prestamos por estado de prestamo
SELECT estado_prestamo, COUNT(*) AS total_prestamos
FROM prestamos 
GROUP BY estado_prestamo
ORDER BY total_prestamos DESC;

-- Obtener el numero total de libros prestados por editorial
SELECT l.editorial, COUNT(p.id_prestamo) AS libros_prestados
FROM libros l
LEFT JOIN prestamos p ON l.id_libro = p.id_libro
WHERE l.editorial IS NOT NULL
GROUP BY l.editorial
ORDER BY libros_prestados DESC;

-- Calcular la multa promedio por usuario para prestamos con estado retraso
SELECT u.nombre, u.primer_apellido, AVG(m.monto) AS multa_promedio
FROM usuarios u
JOIN prestamos p ON u.id_usuario = p.id_usuario
JOIN multas m ON p.id_prestamo = m.id_prestamo
WHERE p.estado_prestamo = 'con retraso'
GROUP BY u.id_usuario, u.nombre, u.primer_apellido
ORDER BY multa_promedio DESC;

-- Mostrar el total de libros disponibles y prestados, agrupados por anio de publicacion
SELECT anio_publicacion,
       SUM(cantidad_total) AS total_libros,
       SUM(cantidad_disponible) AS libros_disponibles,
       SUM(cantidad_total - cantidad_disponible) AS libros_prestados
FROM libros 
WHERE anio_publicacion IS NOT NULL
GROUP BY anio_publicacion
ORDER BY anio_publicacion DESC;

-- Contar la cantidad de usuarios por rango de edad usando CASE y agrupar los resultados
SELECT 
    CASE 
        WHEN EXTRACT(YEAR FROM AGE(fecha_nacimiento)) < 18 THEN 'Menor de 18'
        WHEN EXTRACT(YEAR FROM AGE(fecha_nacimiento)) BETWEEN 18 AND 25 THEN '18-25'
        WHEN EXTRACT(YEAR FROM AGE(fecha_nacimiento)) BETWEEN 26 AND 35 THEN '26-35'
        WHEN EXTRACT(YEAR FROM AGE(fecha_nacimiento)) BETWEEN 36 AND 50 THEN '36-50'
        ELSE 'Mayor de 50'
    END AS rango_edad,
    COUNT(*) AS total_usuarios
FROM usuarios 
WHERE fecha_nacimiento IS NOT NULL
GROUP BY rango_edad
ORDER BY total_usuarios DESC;

-- Calcular el promedio de dias de retraso en devoluciones agrupado por el estado del prestamo
SELECT p.estado_prestamo,
       AVG(d.fecha_devolucion - p.fecha_devolucion_max) AS promedio_dias_retraso
FROM prestamos p
JOIN devoluciones d ON p.id_prestamo = d.id_prestamo
WHERE p.fecha_devolucion_max IS NOT NULL
GROUP BY p.estado_prestamo;

-- Contar la cantidad de libros por clasificacion (de 0 a 5 estrellas)
SELECT clasificacion, COUNT(*) AS total_libros
FROM libros 
GROUP BY clasificacion
ORDER BY clasificacion ASC;

-- Calcular el numero de devoluciones por mes en el ultimo anio
SELECT 
    EXTRACT(YEAR FROM fecha_devolucion) AS anio,
    EXTRACT(MONTH FROM fecha_devolucion) AS mes,
    COUNT(*) AS total_devoluciones
FROM devoluciones 
WHERE fecha_devolucion >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY EXTRACT(YEAR FROM fecha_devolucion), EXTRACT(MONTH FROM fecha_devolucion)
ORDER BY anio DESC, mes DESC;

-- Mostrar el numero de libros prestados en los ultimos 6 meses, agrupados por editorial
SELECT l.editorial, COUNT(p.id_prestamo) AS libros_prestados
FROM libros l
JOIN prestamos p ON l.id_libro = p.id_libro
WHERE p.fecha_prestamo >= CURRENT_DATE - INTERVAL '6 months'
AND l.editorial IS NOT NULL
GROUP BY l.editorial
ORDER BY libros_prestados DESC;

-- Calcular el numero promedio de dias que tarda en devolver los libros cada usuario
SELECT u.nombre, u.primer_apellido,
       AVG(d.fecha_devolucion - p.fecha_prestamo) AS promedio_dias_prestamo
FROM usuarios u
JOIN prestamos p ON u.id_usuario = p.id_usuario
JOIN devoluciones d ON p.id_prestamo = d.id_prestamo
GROUP BY u.id_usuario, u.nombre, u.primer_apellido
ORDER BY promedio_dias_prestamo ASC;