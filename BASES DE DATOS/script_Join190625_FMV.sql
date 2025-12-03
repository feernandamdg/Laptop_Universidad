-- ===================== SUBCONSULTAS =====================

-- 1. Obtener los usuarios cuyo ID sea el mas alto
SELECT * FROM usuarios 
WHERE id_usuario = (SELECT MAX(id_usuario) FROM usuarios);

-- 2. Listar los libros que tengan mas de la cantidad promedio de ejemplares disponibles
SELECT * FROM libros 
WHERE cantidad_disponible > (SELECT AVG(cantidad_disponible) FROM libros);

-- 3. Obtener los usuarios que no tengan ningun prestamo asignado
SELECT * FROM usuarios 
WHERE id_usuario NOT IN (SELECT DISTINCT id_usuario FROM prestamos WHERE id_usuario IS NOT NULL);

-- 4. Listar los libros que tienen menos ejemplares disponibles que la cantidad minima de ejemplares entre todos los libros
-- Insertar libros
INSERT INTO libros (ISBN, titulo, anio_publicacion, editorial, edicion, cantidad_total, cantidad_disponible, clasificacion) VALUES
('978-84-376-0874-9', 'Cien años de soledad', 1967, 'Sudamericana', 1, 5, 3, 5),
('978-84-204-8332-8', 'La casa de los espiritus', 1982, 'Plaza & Janes', 2, 8, 6, 4),
('978-84-204-6777-9', 'La ciudad y los perros', 1963, 'Seix Barral', 1, 3, 1, 4),
('978-84-376-0123-4', 'El laberinto de la soledad', 1950, 'Penguin', 3, 10, 8, 5),
('978-84-204-5555-5', 'La muerte de Artemio Cruz', 1962, 'Fondo de Cultura', 1, 2, 2, 3),
('978-84-376-9999-9', 'Calculo', 2017, 'Preston', 6, 15, 12, 5);
SELECT * FROM libros 
WHERE cantidad_disponible < (SELECT AVG(cantidad_disponible) FROM libros);

-- 5. Seleccionar los prestamos que fueron realizados en el mismo dia en que se registraron los usuarios
-- NO habia entonces inserto datos
INSERT INTO usuarios(nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento, fecha_registro, contrasena, id_rol) VALUES
('Macarena', 'Suarez', 'Lara', 'macarena.suarez@email.com', '5551234398', '1990-05-15', '2025-06-19', 'passMacarena123', 3);
INSERT INTO prestamos (id_usuario, id_libro, fecha_prestamo, fecha_devolucion_max, estado_prestamo) VALUES
(74, 6, '2025-06-19', '2025-06-30', 'pendiente');
SELECT p.* FROM prestamos p
WHERE p.fecha_prestamo = (SELECT u.fecha_registro FROM usuarios u WHERE u.id_usuario = p.id_usuario);

-- 6. Listar los libros cuyo titulo contenga mas caracteres que el titulo del libro con menos caracteres
SELECT * FROM libros 
WHERE LENGTH(titulo) > (SELECT MIN(LENGTH(titulo)) FROM libros);

-- 7. Seleccionar los libros que tengan mas ejemplares disponibles que la cantidad total de un libro especifico (ID 5)
SELECT * FROM libros 
WHERE cantidad_disponible > (SELECT cantidad_total FROM libros WHERE id_libro = 5);

-- 8. Seleccionar los libros cuyo año de publicacion sea mayor que el del libro mas antiguo de una editorial especifica (Penguin)
SELECT * FROM libros 
WHERE anio_publicacion > (SELECT MIN(anio_publicacion) FROM libros WHERE editorial = 'Penguin');

-- 9. Obtener los usuarios cuyo correo electronico sea mas largo que el promedio de longitud de todos los correos
SELECT * FROM usuarios 
WHERE LENGTH(email) > (SELECT AVG(LENGTH(email)) FROM usuarios);

-- 10. Listar los libros cuya edicion sea mayor que la edicion promedio de todos los libros
SELECT * FROM libros 
WHERE edicion > (SELECT AVG(edicion) FROM libros WHERE edicion IS NOT NULL);

-- 11. Seleccionar los prestamos cuya duracion en dias sea mayor que el prestamo con la duracion mas corta
SELECT * FROM prestamos 
WHERE (fecha_devolucion_max - fecha_prestamo) > (
    SELECT MIN(fecha_devolucion_max - fecha_prestamo) 
    FROM prestamos 
    WHERE fecha_devolucion_max IS NOT NULL
);

-- 12. Obtener los usuarios que tienen mas de un prestamo con retraso
-- Insertamos datos porque no devuelve nada
INSERT INTO prestamos (id_usuario, id_libro, fecha_prestamo, fecha_devolucion_max, estado_prestamo) VALUES
(1, 4, '2024-03-10', '2024-03-24', 'con retraso'),
(1, 5, '2024-03-15', '2024-03-29', 'con retraso');
SELECT * FROM usuarios 
WHERE id_usuario IN (
    SELECT id_usuario 
    FROM prestamos 
    WHERE estado_prestamo = 'con retraso' 
    GROUP BY id_usuario 
    HAVING COUNT(*) > 1
);

-- 13. Seleccionar los prestamos cuya fecha de prestamo sea anterior a la fecha de registro del usuario
SELECT p.* FROM prestamos p
WHERE p.fecha_prestamo < (SELECT u.fecha_registro FROM usuarios u WHERE u.id_usuario = p.id_usuario);

-- ===================== CONSULTAS CON UNION =====================

-- 1. Clasificacion de libros por calificacion
SELECT 
    titulo AS "Titulo",
    editorial AS "Editorial", 
    edicion AS "Edicion",
    TO_CHAR(TO_DATE(anio_publicacion::text, 'YYYY'), 'YYYY-MM-DD') AS "Fecha_Publicacion",
    CASE 
        WHEN clasificacion = 0 THEN 'No Recomendado'
        WHEN clasificacion = 1 THEN 'Pesimo'
        WHEN clasificacion = 2 THEN 'Evitar'
        WHEN clasificacion = 3 THEN 'Regular'
        WHEN clasificacion = 4 THEN 'Lectura Entretenida'
        WHEN clasificacion = 5 THEN 'Excelente'
        ELSE 'Sin Calificacion'
    END AS "Clasificacion"
FROM libros
WHERE clasificacion = 0
UNION
SELECT 
    titulo AS "Titulo",
    editorial AS "Editorial", 
    edicion AS "Edicion",
    TO_CHAR(TO_DATE(anio_publicacion::text, 'YYYY'), 'YYYY-MM-DD') AS "Fecha_Publicacion",
    CASE 
        WHEN clasificacion = 0 THEN 'No Recomendado'
        WHEN clasificacion = 1 THEN 'Pesimo'
        WHEN clasificacion = 2 THEN 'Evitar'
        WHEN clasificacion = 3 THEN 'Regular'
        WHEN clasificacion = 4 THEN 'Lectura Entretenida'
        WHEN clasificacion = 5 THEN 'Excelente'
        ELSE 'Sin Calificacion'
    END AS "Clasificacion"
FROM libros
WHERE clasificacion = 5;

-- 2. Clasificacion de usuarios por antiguedad
SELECT 
    CONCAT(nombre, ' ', primer_apellido, ' ', COALESCE(segundo_apellido, '')) AS "Nombre_Completo",
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) AS "Edad",
    fecha_registro AS "Fecha_Registro",
    CASE 
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) < 1 THEN 
            CONCAT('Recien ', (SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol))
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) BETWEEN 1 AND 3 THEN 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' con conocimiento del sistema')
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) BETWEEN 3 AND 4 THEN 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' con experiencia')
        ELSE 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' Veterano')
    END AS "Informacion"
FROM usuarios
WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) < 1
UNION
SELECT 
    CONCAT(nombre, ' ', primer_apellido, ' ', COALESCE(segundo_apellido, '')) AS "Nombre_Completo",
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) AS "Edad",
    fecha_registro AS "Fecha_Registro",
    CASE 
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) < 1 THEN 
            CONCAT('Recien ', (SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol))
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) BETWEEN 1 AND 3 THEN 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' con conocimiento del sistema')
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) BETWEEN 3 AND 4 THEN 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' con experiencia')
        ELSE 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' Veterano')
    END AS "Informacion"
FROM usuarios
WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) >= 5;

-- --------------------- CONSULTAS CON CASE ---------------------------

-- Replica de ejercicio 1 con CASE
SELECT 
    titulo AS "Titulo",
    editorial AS "Editorial", 
    edicion AS "Edicion",
    CASE 
        WHEN anio_publicacion IS NOT NULL THEN TO_CHAR(TO_DATE(anio_publicacion::text, 'YYYY'), 'YYYY-MM-DD')
        ELSE 'Sin fecha'
    END AS "Fecha_Publicacion",
    CASE 
        WHEN clasificacion = 0 THEN 'No Recomendado'
        WHEN clasificacion = 1 THEN 'Pesimo'
        WHEN clasificacion = 2 THEN 'Evitar'
        WHEN clasificacion = 3 THEN 'Regular'
        WHEN clasificacion = 4 THEN 'Lectura Entretenida'
        WHEN clasificacion = 5 THEN 'Excelente'
        ELSE 'Sin Calificacion'
    END AS "Clasificacion"
FROM libros;

-- Replica de ejercicio 2 con CASE
SELECT 
    CONCAT(nombre, ' ', primer_apellido, ' ', COALESCE(segundo_apellido, '')) AS "Nombre_Completo",
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) AS "Edad",
    fecha_registro AS "Fecha_Registro",
    CASE 
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) < 1 THEN 
            CONCAT('Recien ', (SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol))
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) BETWEEN 1 AND 3 THEN 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' con conocimiento del sistema')
        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_registro)) BETWEEN 3 AND 4 THEN 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' con experiencia')
        ELSE 
            CONCAT((SELECT nombre_rol FROM roles WHERE id_rol = usuarios.id_rol), ' Veterano')
    END AS "Informacion"
FROM usuarios;

-- ===================== CONSULTAS CON JOIN =====================

-- 1. Lista completa de todos los usuarios y los libros que han prestado
SELECT u.nombre, u.primer_apellido, l.titulo, p.fecha_prestamo, p.estado_prestamo
FROM usuarios u
LEFT JOIN prestamos p ON u.id_usuario = p.id_usuario
LEFT JOIN libros l ON p.id_libro = l.id_libro;

-- 2. Todos los libros y los autores correspondientes, incluyendo libros sin autor
SELECT l.titulo, l.editorial, a.nombre_autor, a.apellido_paterno, a.apell ido_materno
FROM libros l
LEFT JOIN libroautor la ON l.id_libro = la.id_libro
LEFT JOIN autores a ON la.id_autor = a.id_autor;

-- 3. Lista de todos los autores y los libros que han escrito
SELECT a.nombre_autor, a.apellido_paterno, a.apellido_materno, l.titulo, l.editorial
FROM autores a
LEFT JOIN libroautor la ON a.id_autor = la.id_autor
LEFT JOIN libros l ON la.id_libro = l.id_libro;

-- 4. Todas las devoluciones con prestamos correspondientes, incluyendo prestamos no devueltos
SELECT p.id_prestamo, p.fecha_prestamo, p.estado_prestamo, d.fecha_devolucion
FROM prestamos p
LEFT JOIN devoluciones d ON p.id_prestamo = d.id_prestamo;

-- 5. Lista de usuarios con y sin prestamos, mostrando detalles de prestamo si existen
SELECT u.nombre, u.primer_apellido, u.email, p.id_prestamo, p.fecha_prestamo, p.estado_prestamo
FROM usuarios u
LEFT JOIN prestamos p ON u.id_usuario = p.id_usuario;

-- 6. Todas las editoriales y los libros publicados por ellas
SELECT DISTINCT l1.editorial, l2.titulo, l2.anio_publicacion
FROM (SELECT DISTINCT editorial FROM libros WHERE editorial IS NOT NULL) l1
LEFT JOIN libros l2 ON l1.editorial = l2.editorial;

-- 7. Todos los libros prestados, incluyendo autores y libros no prestados
SELECT l.titulo, l.editorial, a.nombre_autor, a.apellido_paterno, p.fecha_prestamo, p.estado_prestamo
FROM libros l
LEFT JOIN prestamos p ON l.id_libro = p.id_libro
LEFT JOIN libroautor la ON l.id_libro = la.id_libro
LEFT JOIN autores a ON la.id_autor = a.id_autor;

-- 8. Todos los prestamos realizados y el nombre del usuario que lo hizo
SELECT p.id_prestamo, p.fecha_prestamo, p.estado_prestamo, u.nombre, u.primer_apellido, l.titulo
FROM prestamos p
INNER JOIN usuarios u ON p.id_usuario = u.id_usuario
INNER JOIN libros l ON p.id_libro = l.id_libro;

-- 9. Todos los libros disponibles, incluyendo aquellos sin autor o no prestados
SELECT l.titulo, l.cantidad_disponible, a.nombre_autor, a.apellido_paterno, 
       CASE WHEN p.id_prestamo IS NULL THEN 'No prestado' ELSE 'Prestado' END AS estado_prestamo
FROM libros l
LEFT JOIN libroautor la ON l.id_libro = la.id_libro
LEFT JOIN autores a ON la.id_autor = a.id_autor
LEFT JOIN prestamos p ON l.id_libro = p.id_libro AND p.estado_prestamo = 'pendiente';

-- 10. Todas las multas de prestamos, incluyendo usuarios que las generaron y prestamos sin multa
SELECT p.id_prestamo, u.nombre, u.primer_apellido, m.monto, m.pagado, m.fecha_multa
FROM prestamos p
LEFT JOIN multas m ON p.id_prestamo = m.id_prestamo
INNER JOIN usuarios u ON p.id_usuario = u.id_usuario;