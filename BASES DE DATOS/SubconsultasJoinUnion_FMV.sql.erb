-- 1. FUNCIONES DE CADENAS

-- Muestra solo a las personas que su nombre tiene mas de 5 letras
SELECT * FROM usuarios 
WHERE LENGTH(nombre) > 5;

-- Muestra a las personas que su nombre tiene entre 5 y 7 caracteres
SELECT * FROM usuarios 
WHERE LENGTH(nombre) BETWEEN 5 AND 7;

-- Muestra a las personas que su nombre tiene mas de 7 caracteres y alguno de sus apellidos tenga entre 5 y 7 caracteres
SELECT * FROM usuarios 
WHERE LENGTH(nombre) > 7 
AND (LENGTH(primer_apellido) BETWEEN 5 AND 7 OR LENGTH(segundo_apellido) BETWEEN 5 AND 7);

-- Muestra los primeros tres caracteres del nombre
SELECT nombre, LEFT(nombre, 3) AS primeros_tres_caracteres
FROM usuarios;

-- Muestra los ultimos 3 caracteres del nombre
SELECT nombre, RIGHT(nombre, 3) AS ultimos_tres_caracteres
FROM usuarios;

-- Muestra del 2do al 5to caracter del nombre
SELECT nombre, SUBSTRING(nombre, 2, 4) AS caracteres_2_al_5
FROM usuarios;

-- Reemplaza las d por s en nombre
SELECT nombre, REPLACE(nombre, 'd', 's') AS nombre_con_s
FROM usuarios;

-- Obten la longitud del apellido paterno
SELECT primer_apellido, LENGTH(primer_apellido) AS longitud_apellido_paterno
FROM usuarios;

-- Muestra en mayusculas el nombre
SELECT nombre, UPPER(nombre) AS nombre_mayusculas
FROM usuarios;

-- Muestra en minusculas el apellido paterno
SELECT primer_apellido, LOWER(primer_apellido) AS apellido_minusculas
FROM usuarios;

-- Muestra el nombre completo empezando por el apellido paterno con mayusculas en una sola columna
SELECT UPPER(primer_apellido) || ' ' || 
       COALESCE(segundo_apellido || ' ', '') || 
       nombre AS nombre_completo
FROM usuarios;

-- Muestra el nombre de las personas con las E reemplazadas con el numero 3
SELECT nombre, REPLACE(REPLACE(nombre, 'E', '3'), 'e', '3') AS nombre_con_3
FROM usuarios;

-- Muestra el nombre completo de las personas con las o reemplazados con el numero 0 en una sola columna
SELECT REPLACE(REPLACE(nombre || ' ' || primer_apellido || ' ' || COALESCE(segundo_apellido, ''), 'O', '0'), 'o', '0') AS nombre_completo_con_0
FROM usuarios;

-- Libros con edicion entre 1 y 5 y cuyo titulo comience con letra mayuscula
SELECT * FROM libros 
WHERE edicion BETWEEN 1 AND 5 
AND titulo ~ '^[A-Z]';

-- Usuarios cuyo segundo apellido sea NULL y cuyo primer apellido tenga mas de 6 caracteres
SELECT * FROM usuarios 
WHERE segundo_apellido IS NULL 
AND LENGTH(primer_apellido) > 6;

-- Muestra el nombre de las personas reemplazando los siguientes caracteres A-->@,E-->3,I-->!, O-->0
SELECT nombre,
       REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(nombre, 'A', '@'), 'a', '@'), 'E', '3'), 'e', '3'), 'I', '!'), 'i', '!'), 'O', '0'), 'o', '0') AS nombre_codificado
FROM usuarios;

-- Convierte los primeros tres caracteres del nombre en mayuscula
SELECT nombre,
       UPPER(LEFT(nombre, 3)) || SUBSTRING(nombre, 4) AS nombre_modificado
FROM usuarios
WHERE LENGTH(nombre) >= 3;

-- Convierte el ultimo caracter del nombre en mayusculas
SELECT nombre,
       LEFT(nombre, LENGTH(nombre) - 1) || UPPER(RIGHT(nombre, 1)) AS nombre_modificado
FROM usuarios
WHERE LENGTH(nombre) >= 1;

-- Convierte el 3er caracter del nombre en Mayuscula
SELECT nombre,
       LEFT(nombre, 2) || UPPER(SUBSTRING(nombre, 3, 1)) || SUBSTRING(nombre, 4) AS nombre_modificado
FROM usuarios
WHERE LENGTH(nombre) >= 3;

-- Convierte el 2do y 4to caracter del nombre a Mayuscula
SELECT nombre,
       LEFT(nombre, 1) || 
       UPPER(SUBSTRING(nombre, 2, 1)) || 
       SUBSTRING(nombre, 3, 1) || 
       UPPER(SUBSTRING(nombre, 4, 1)) || 
       SUBSTRING(nombre, 5) AS nombre_modificado
FROM usuarios
WHERE LENGTH(nombre) >= 4;

-- Convierte a mayusculas el segundo y ultimo caracter
SELECT nombre,
       LEFT(nombre, 1) || 
       UPPER(SUBSTRING(nombre, 2, 1)) || 
       SUBSTRING(nombre, 3, LENGTH(nombre) - 3) || 
       UPPER(RIGHT(nombre, 1)) AS nombre_modificado
FROM usuarios
WHERE LENGTH(nombre) >= 2;

-- Convierte a mayuscula el segundo, cuarto y penultimo caracter del nombre
SELECT nombre,
       LEFT(nombre, 1) || 
       UPPER(SUBSTRING(nombre, 2, 1)) || 
       SUBSTRING(nombre, 3, 1) || 
       UPPER(SUBSTRING(nombre, 4, 1)) || 
       SUBSTRING(nombre, 5, LENGTH(nombre) - 6) || 
       UPPER(SUBSTRING(nombre, LENGTH(nombre) - 1, 1)) || 
       RIGHT(nombre, 1) AS nombre_modificado
FROM usuarios
WHERE LENGTH(nombre) >= 4;

-- Selecciona los usuarios con correo electronico registrado, pero cuya direccion de email no incluya mas de 10 caracteres antes del simbolo @
SELECT * FROM usuarios 
WHERE email IS NOT NULL 
AND LENGTH(SPLIT_PART(email, '@', 1)) <= 10;

-- Usuarios cuyo nombre tenga exactamente dos palabras y cuyo rol este entre 'administrador' y 'bibliotecario'
SELECT u.* FROM usuarios u
JOIN roles r ON u.id_rol = r.id_rol
WHERE LENGTH(nombre) - LENGTH(REPLACE(nombre, ' ', '')) = 1
AND r.nombre_rol IN ('administrador', 'bibliotecario');

-- Usuarios cuyo nombre completo tenga exactamente 30 caracteres, y cuya contrasena tenga al menos 8 caracteres
SELECT * FROM usuarios 
WHERE LENGTH(nombre || ' ' || primer_apellido || ' ' || COALESCE(segundo_apellido, '')) = 30
AND LENGTH(contrasena) >= 8;

-- 2. INSTRUCCIONES LIKE/SIMILAR TO

-- Las personas que se llamen Eduardo sin importar que tengan 2 nombres
SELECT * FROM usuarios 
WHERE nombre LIKE '%Eduardo%';

-- Las personas que su segundo caracter sea una "d"
SELECT * FROM usuarios 
WHERE nombre LIKE '_d%';

-- Los que no empiecen su nombre con una vocal
SELECT * FROM usuarios 
WHERE nombre NOT SIMILAR TO '[AEIOUaeiou]%';

-- Los que empiecen su nombre con una vocal y terminen con s
SELECT * FROM usuarios 
WHERE nombre SIMILAR TO '[AEIOUaeiou]%[sS]';
-- Los que su tercer caracter del nombre sea una G
-- no habia entonces insertar
INSERT INTO usuarios (nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento, id_rol, contrasena)
VALUES
('Logan', 'Mercuri', 'Vega', 'logan.mercuri@example.com', '5598269876', '1991-08-13', 3, 'PassLogan123');
SELECT * FROM usuarios 
WHERE nombre SIMILAR TO '__[Gg]%';

-- Los que su primer caracter en el apellido paterno sea 'E' y el 4 sea 'A'
--NO HABIA entonces insertar
INSERT INTO usuarios (nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento, id_rol, contrasena)
VALUES
('Morgan', 'Estaño', 'Hull', 'morgan.estaño@example.com', '5998269876', '1999-09-17', 3, 'PassMorgan123');
SELECT * FROM usuarios 
WHERE primer_apellido SIMILAR TO 'E__[Aa]%';

-- Los que tengan por lo menos una 'E' en su nombre
SELECT * FROM usuarios 
WHERE nombre LIKE '%E%' OR nombre LIKE '%e%';

-- Los que se llaman Eduardo y Cualquiera de sus apellidos empiece con 'C'
--NO HABIA entonces insertar
INSERT INTO usuarios (nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento, id_rol, contrasena)
VALUES
('Eduardo', 'Cobalto', 'Bautista', 'eduardo.cobalto@example.com', '5998288876', '2000-09-10', 3, 'PassEduardo123');
SELECT * FROM usuarios 
WHERE nombre LIKE '%Eduardo%' 
AND (primer_apellido LIKE 'C%' OR segundo_apellido LIKE 'C%');

-- Las personas que su apellido materno empiece con la primera mitad del alfabeto [A-M] pero que no empiecen ni con A ni con E
SELECT * FROM usuarios 
WHERE segundo_apellido SIMILAR TO '[B-DF-Mb-df-m]%';

-- Las personas que su apellido paterno empiece con la segunda mitad del alfabeto [N-Z]
SELECT * FROM usuarios 
WHERE primer_apellido SIMILAR TO '[N-Zn-z]%';

-- Obtener los libros cuyo titulo contenga la palabra "Historia"
SELECT * FROM libros 
WHERE titulo LIKE '%Historia%';

-- Listar los usuarios cuyo apellido paterno termine con "ez"
SELECT * FROM usuarios 
WHERE primer_apellido LIKE '%ez';

-- Seleccionar todos los usuarios cuyo correo electronico sea de Gmail
--No habia entonces inserto
INSERT INTO usuarios (nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento, id_rol, contrasena)
VALUES
('Fernando', 'Farias', 'Villanueva', 'fernando.farias@gmail.com', '5982288876', '2006-12-26', 3, 'PassFernando123');
SELECT * FROM usuarios 
WHERE email LIKE '%@gmail.com';

-- Consultar los libros cuya editorial contenga la palabra "Editores"
--No habia entonces inserto
INSERT INTO libros (isbn, titulo, anio_publicacion, editorial, edicion, cantidad_total, cantidad_disponible, clasificacion)
VALUES
('978-3-16-118410-0', 'Camino a la luz', 2005, 'Editores Unidos', 2, 20, 13, 5);
SELECT * FROM libros 
WHERE editorial LIKE '%Editores%';

-- Listar los usuarios cuyos nombres contengan una vocal seguida de una "n"
SELECT * FROM usuarios 
WHERE nombre SIMILAR TO '%[AEIOUaeiou]n%';

-- Obtener todos los usuarios cuyos numeros de telefono terminen en "00"
--No hay enonces inserto
INSERT INTO usuarios (nombre, primer_apellido, segundo_apellido, email, telefono, fecha_nacimiento, id_rol, contrasena)
VALUES
('Rosa', 'Oliva', 'Oliva', 'rosa.oliva@gmail.com', '5982280000', '1993-12-13', 3, 'PassRosa123');
SELECT * FROM usuarios 
WHERE telefono LIKE '%00';

-- Consultar los autores cuyo nombre contenga la letra "J" en cualquier parte
SELECT * FROM autores 
WHERE nombre_autor LIKE '%J%' OR nombre_autor LIKE '%j%';

-- Obtener los usuarios cuyo nombre contenga exactamente cinco caracteres
SELECT * FROM usuarios 
WHERE nombre SIMILAR TO '_____';

-- Obtener los titulos de libros que contengan numeros en cualquier parte
SELECT * FROM libros 
WHERE titulo SIMILAR TO '%[0-9]%';