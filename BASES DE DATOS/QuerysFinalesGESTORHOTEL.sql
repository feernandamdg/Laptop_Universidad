INSERT INTO uso_spa (id_reserva, id_tratamiento, hora_inicio, hora_fin) VALUES
(1, 1, '2025-07-02 10:00:00', '2025-07-02 11:00:00'),
(2, 2, '2025-07-11 14:30:00', '2025-07-11 15:30:00'),
(3, 3, '2025-07-04 09:00:00', '2025-07-04 10:30:00'),
(4, 4, '2025-07-09 12:00:00', '2025-07-09 13:00:00'),
(5, 5, '2025-07-06 11:00:00', '2025-07-06 12:30:00'),
(6, 1, '2025-07-16 17:00:00', '2025-07-16 18:00:00'),
(7, 2, '2025-07-22 15:00:00', '2025-07-22 16:00:00'),
(8, 3, '2025-07-13 13:00:00', '2025-07-13 14:00:00'),
(9, 4, '2025-07-18 16:30:00', '2025-07-18 17:30:00'),
(10, 5, '2025-07-24 10:30:00', '2025-07-24 11:30:00'),
(11, 1, '2025-07-05 09:00:00', '2025-07-05 10:00:00'),
(12, 2, '2025-07-09 16:00:00', '2025-07-09 17:00:00'),
(13, 3, '2025-07-07 14:00:00', '2025-07-07 15:00:00'),
(14, 4, '2025-07-14 11:00:00', '2025-07-14 12:00:00'),
(15, 5, '2025-07-19 10:00:00', '2025-07-19 11:30:00'),
(16, 1, '2025-07-25 12:00:00', '2025-07-25 13:00:00'),
(17, 2, '2025-07-08 17:00:00', '2025-07-08 18:00:00'),
(18, 3, '2025-07-15 13:30:00', '2025-07-15 14:30:00'),
(19, 4, '2025-07-21 11:00:00', '2025-07-21 12:30:00'),
(20, 5, '2025-07-03 15:00:00', '2025-07-03 16:00:00'),
(21, 1, '2025-07-10 09:30:00', '2025-07-10 10:30:00'),
(22, 2, '2025-07-12 14:00:00', '2025-07-12 15:00:00'),
(23, 3, '2025-07-16 10:00:00', '2025-07-16 11:00:00'),
(24, 4, '2025-07-22 13:00:00', '2025-07-22 14:00:00'),
(25, 5, '2025-07-30 15:30:00', '2025-07-30 16:30:00'),
(26, 1, '2025-07-05 11:30:00', '2025-07-05 12:30:00'),
(27, 2, '2025-07-08 12:00:00', '2025-07-08 13:00:00'),
(28, 3, '2025-07-10 16:00:00', '2025-07-10 17:00:00'),
(29, 4, '2025-07-14 15:00:00', '2025-07-14 16:00:00'),
(30, 5, '2025-07-20 14:30:00', '2025-07-20 15:30:00'),
(31, 1, '2025-07-28 10:00:00', '2025-07-28 11:00:00'),
(32, 2, '2025-07-13 17:00:00', '2025-07-13 18:00:00'),
(33, 3, '2025-07-25 10:00:00', '2025-07-25 11:00:00'),
(34, 4, '2025-07-18 12:30:00', '2025-07-18 13:30:00'),
(35, 5, '2025-07-26 15:00:00', '2025-07-26 16:00:00'),
(36, 1, '2025-07-06 13:00:00', '2025-07-06 14:00:00'),
(37, 2, '2025-07-23 11:30:00', '2025-07-23 12:30:00'),
(38, 3, '2025-07-29 12:00:00', '2025-07-29 13:00:00'),
(39, 4, '2025-07-31 16:00:00', '2025-07-31 17:00:00'),
(40, 5, '2025-07-27 10:30:00', '2025-07-27 11:30:00');

-- RECIEN AGREGADO PARA PODER REALIZAR CONSULTAS DADAS EN CLASE
INSERT INTO uso_gimnasio (id_reserva, hora_entrada, hora_salida) VALUES
(1,  '2025-07-01 07:00:00', '2025-07-01 08:15:00'),
(2,  '2025-07-11 09:30:00', '2025-07-11 10:45:00'),
(3,  '2025-07-04 08:00:00', '2025-07-04 09:00:00'),
(4,  '2025-07-08 06:30:00', '2025-07-08 07:45:00'),
(5,  '2025-07-06 07:00:00', '2025-07-06 08:00:00'),
(6,  '2025-07-16 18:00:00', '2025-07-16 19:30:00'),
(7,  '2025-07-21 08:30:00', '2025-07-21 09:30:00'),
(8,  '2025-07-13 07:00:00', '2025-07-13 08:30:00'),
(9,  '2025-07-18 06:00:00', '2025-07-18 07:00:00'),
(10, '2025-07-23 10:00:00', '2025-07-23 11:15:00'),
(11, '2025-07-05 07:15:00', '2025-07-05 08:00:00'),
(12, '2025-07-09 07:00:00', '2025-07-09 08:15:00'),
(13, '2025-07-07 07:00:00', '2025-07-07 08:00:00'),
(14, '2025-07-14 08:00:00', '2025-07-14 09:00:00'),
(15, '2025-07-19 06:45:00', '2025-07-19 08:00:00'),
(16, '2025-07-24 10:00:00', '2025-07-24 11:00:00'),
(17, '2025-07-28 08:00:00', '2025-07-28 09:15:00'),
(18, '2025-07-08 07:00:00', '2025-07-08 08:00:00'),
(19, '2025-07-15 06:30:00', '2025-07-15 07:30:00'),
(20, '2025-07-20 09:00:00', '2025-07-20 10:15:00'),
(21, '2025-07-03 08:00:00', '2025-07-03 09:00:00'),
(22, '2025-07-11 06:00:00', '2025-07-11 07:15:00'),
(23, '2025-07-17 07:30:00', '2025-07-17 09:00:00'),
(24, '2025-07-22 08:00:00', '2025-07-22 09:15:00'),
(25, '2025-07-29 06:30:00', '2025-07-29 08:00:00'),
(26, '2025-07-01 07:15:00', '2025-07-01 08:15:00'),
(27, '2025-07-06 08:00:00', '2025-07-06 09:00:00'),
(28, '2025-07-09 10:00:00', '2025-07-09 11:00:00'),
(29, '2025-07-11 07:30:00', '2025-07-11 09:00:00'),
(30, '2025-07-14 06:00:00', '2025-07-14 07:30:00'),
(31, '2025-07-21 08:00:00', '2025-07-21 09:00:00'),
(32, '2025-07-23 10:15:00', '2025-07-23 11:45:00'),
(33, '2025-07-26 06:30:00', '2025-07-26 07:45:00'),
(34, '2025-07-04 07:00:00', '2025-07-04 08:30:00'),
(35, '2025-07-10 07:15:00', '2025-07-10 08:15:00'),
(36, '2025-07-13 06:45:00', '2025-07-13 07:45:00'),
(37, '2025-07-17 07:30:00', '2025-07-17 09:00:00'),
(38, '2025-07-20 08:00:00', '2025-07-20 09:30:00'),
(39, '2025-07-27 07:00:00', '2025-07-27 08:00:00'),
(40, '2025-07-07 10:00:00', '2025-07-07 11:00:00'),
(41, '2025-07-16 06:00:00', '2025-07-16 07:30:00'),
(42, '2025-07-23 09:30:00', '2025-07-23 11:00:00'),
(43, '2025-07-29 08:00:00', '2025-07-29 09:00:00'),
(44, '2025-07-13 07:30:00', '2025-07-13 08:30:00'),
(45, '2025-07-15 06:00:00', '2025-07-15 07:15:00'),
(46, '2025-07-20 09:00:00', '2025-07-20 10:00:00'),
(47, '2025-07-22 08:15:00', '2025-07-22 09:15:00'),
(48, '2025-07-26 07:45:00', '2025-07-26 09:00:00'),
(49, '2025-07-28 08:30:00', '2025-07-28 09:30:00'),
(50, '2025-07-29 10:00:00', '2025-07-29 11:15:00');

INSERT INTO uso_transporte (id_reserva, fecha_hora_salida, fecha_hora_regreso, duracion_horas) VALUES
(1, '2025-07-02 09:00:00', '2025-07-02 11:30:00', 2.50),
(2, '2025-07-11 13:00:00', '2025-07-11 15:00:00', 2.00),
(3, '2025-07-04 07:30:00', '2025-07-04 09:00:00', 1.50),
(5, '2025-07-06 12:00:00', '2025-07-06 14:30:00', 2.50),
(6, '2025-07-16 10:00:00', '2025-07-16 11:15:00', 1.25),
(7, '2025-07-21 17:00:00', '2025-07-21 18:30:00', 1.50),
(8, '2025-07-13 09:00:00', '2025-07-13 11:00:00', 2.00),
(9, '2025-07-18 08:30:00', '2025-07-18 10:00:00', 1.50),
(11, '2025-07-05 16:00:00', '2025-07-05 17:45:00', 1.75),
(12, '2025-07-09 11:00:00', '2025-07-09 13:00:00', 2.00),
(13, '2025-07-07 15:00:00', '2025-07-07 16:30:00', 1.50),
(14, '2025-07-14 10:00:00', '2025-07-14 12:00:00', 2.00),
(15, '2025-07-19 14:00:00', '2025-07-19 15:00:00', 1.00),
(17, '2025-07-27 10:30:00', '2025-07-27 12:30:00', 2.00),
(18, '2025-07-08 13:00:00', '2025-07-08 14:30:00', 1.50),
(19, '2025-07-15 09:00:00', '2025-07-15 10:00:00', 1.00),
(20, '2025-07-21 08:00:00', '2025-07-21 09:15:00', 1.25),
(22, '2025-07-11 15:00:00', '2025-07-11 17:00:00', 2.00),
(23, '2025-07-16 07:45:00', '2025-07-16 09:30:00', 1.75),
(24, '2025-07-22 11:30:00', '2025-07-22 13:30:00', 2.00),
(25, '2025-07-30 12:00:00', '2025-07-30 14:30:00', 2.50),
(26, '2025-07-01 07:00:00', '2025-07-01 08:30:00', 1.50),
(28, '2025-07-09 14:00:00', '2025-07-09 15:45:00', 1.75),
(30, '2025-07-14 15:30:00', '2025-07-14 17:30:00', 2.00),
(32, '2025-07-23 07:30:00', '2025-07-23 09:00:00', 1.50),
(34, '2025-07-04 13:00:00', '2025-07-04 14:30:00', 1.50),
(35, '2025-07-10 08:00:00', '2025-07-10 09:45:00', 1.75),
(38, '2025-07-20 17:00:00', '2025-07-20 19:00:00', 2.00),
(40, '2025-07-07 13:00:00', '2025-07-07 15:30:00', 2.50),
(43, '2025-07-29 07:00:00', '2025-07-29 09:00:00', 2.00);

INSERT INTO ticket_restaurante (id_reserva, hora_entrada, hora_salida) VALUES
(1,  '2025-07-02 12:30:00', '2025-07-02 13:45:00'),
(3,  '2025-07-04 18:00:00', '2025-07-04 19:15:00'),
(5,  '2025-07-06 08:30:00', '2025-07-06 09:40:00'),
(7,  '2025-07-21 13:00:00', '2025-07-21 14:00:00'),
(8,  '2025-07-13 19:30:00', '2025-07-13 21:00:00'),
(9,  '2025-07-18 12:00:00', '2025-07-18 13:15:00'),
(12, '2025-07-10 07:45:00', '2025-07-10 09:00:00'),
(14, '2025-07-14 13:15:00', '2025-07-14 14:20:00'),
(15, '2025-07-19 17:45:00', '2025-07-19 18:50:00'),
(17, '2025-07-28 20:30:00', '2025-07-28 22:00:00'),
(18, '2025-07-10 12:30:00', '2025-07-10 13:30:00'),
(19, '2025-07-15 14:00:00', '2025-07-15 15:00:00'),
(22, '2025-07-12 09:00:00', '2025-07-12 10:15:00'),
(23, '2025-07-17 18:00:00', '2025-07-17 19:00:00'),
(25, '2025-07-30 12:30:00', '2025-07-30 13:30:00'),
(26, '2025-07-02 18:45:00', '2025-07-02 19:30:00'),
(28, '2025-07-09 14:00:00', '2025-07-09 15:10:00'),
(30, '2025-07-14 19:00:00', '2025-07-14 20:30:00'),
(34, '2025-07-04 08:00:00', '2025-07-04 09:00:00'),
(35, '2025-07-10 18:30:00', '2025-07-10 19:40:00');

INSERT INTO detalle_ticket (id_ticket, id_platillo, cantidad) VALUES
(1,  1,  2), (1, 17, 1),
(2,  6,  1), (2, 29, 2),
(3,  3,  1), (3, 18, 2), (3, 35, 1),
(4,  8,  1), (4, 32, 2),
(5, 10, 1), (5, 13, 1), (5, 34, 1),
(6, 22, 2), (6, 37, 1),
(7,  5,  1), (7, 19, 2),
(8, 28, 1), (8, 30, 1), (8, 15, 2),
(9,  7,  1), (9, 12, 1),
(10, 31, 2), (10, 20, 1), (10, 38, 1),
(11, 26, 2), (11, 39, 2),
(12, 14, 1), (12, 4, 1),
(13, 23, 2), (13, 16, 1),
(14,  9,  1), (14, 11, 1), (14, 40, 1),
(15, 41, 1), (15, 2, 1), (15, 24, 2),
(16, 25, 1), (16, 19, 2),
(17, 27, 2), (17, 44, 1),
(18, 45, 1), (18, 36, 2), (18, 21, 1),
(19, 33, 1), (19, 43, 2),
(20, 10, 1), (20, 13, 1), (20, 42, 1);


-- ------------------- 1.	Una vista que permita visualizar las habitaciones, su tipo, y la información -------------------
-- -------------------      correspondiente a su reservación.                                           ------------------
CREATE OR REPLACE VIEW habitaciones_con_reservas AS
SELECT
    ha.id_habitacion,
    ha.numero AS numero_habitacion,
    ha.piso,
    ha.tipo,
    ha.precio_por_noche,
    ha.estado AS estado_habitacion,
    r.id_reserva,
    r.fecha_entrada,
    r.fecha_salida,
    r.numero_personas,
    r.metodo_pago,
    r.estado AS estado_reserva,
    h.nombre_completo AS nombre_huesped
FROM habitacion ha
LEFT JOIN reserva r ON ha.id_habitacion = r.id_habitacion
LEFT JOIN huesped h ON r.id_huesped = h.id_huesped
ORDER BY ha.numero;

SELECT * FROM habitaciones_con_reservas
--para visualizar de acuerdo a su estado
WHERE estado_reserva = 'Activa';


-- -------------------- 2. Obtener la cantidad de habitaciones por tipo de habitaciones ------------------------------
-- creación de la funcion 
CREATE OR REPLACE FUNCTION obtener_cantidad_por_tipo()
RETURNS TABLE(tipo_habitacion VARCHAR, cantidad INTEGER) AS
$$
BEGIN
    RETURN QUERY
    SELECT tipo, COUNT(*)::INTEGER AS cantidad
    FROM habitacion
    GROUP BY tipo
    ORDER BY tipo;
END;
$$ LANGUAGE plpgsql;

	-- ejecucion
	SELECT * FROM obtener_cantidad_por_tipo();

-- 3.	Una vista que permita visualizar los servicios adicionales que contrató cada uno de los huéspedes. ---------------
CREATE OR REPLACE VIEW vista_servicios_contratados_por_huesped AS
SELECT
    h.id_huesped,
    h.nombre_completo,
    sa.nombre AS servicio,
    'Spa' AS tipo_servicio,
    us.hora_inicio,
    us.hora_fin
FROM huesped h
JOIN reserva r ON h.id_huesped = r.id_huesped
JOIN uso_spa us ON r.id_reserva = us.id_reserva
JOIN tratamiento t ON us.id_tratamiento = t.id_tratamiento
JOIN spa s ON t.id_servicio = s.id_servicio
JOIN servicios_adicionales sa ON s.id_servicio = sa.id_servicio

UNION

SELECT
    h.id_huesped,
    h.nombre_completo,
    sa.nombre AS servicio,
    'Gimnasio' AS tipo_servicio,
    ug.hora_entrada AS hora_inicio,
    ug.hora_salida AS hora_fin
FROM huesped h
JOIN reserva r ON h.id_huesped = r.id_huesped
JOIN uso_gimnasio ug ON r.id_reserva = ug.id_reserva
JOIN gimnasio g ON g.id_servicio = (SELECT id_servicio FROM servicios_adicionales WHERE nombre ILIKE 'Gimnasio' LIMIT 1)
JOIN servicios_adicionales sa ON g.id_servicio = sa.id_servicio

UNION

SELECT
    h.id_huesped,
    h.nombre_completo,
    sa.nombre AS servicio,
    'Transporte' AS tipo_servicio,
    ut.fecha_hora_salida AS hora_inicio,
    ut.fecha_hora_regreso AS hora_fin
FROM huesped h
JOIN reserva r ON h.id_huesped = r.id_huesped
JOIN uso_transporte ut ON r.id_reserva = ut.id_reserva
JOIN transporte t ON t.id_servicio = (SELECT id_servicio FROM servicios_adicionales WHERE nombre ILIKE 'Transporte' LIMIT 1)
JOIN servicios_adicionales sa ON t.id_servicio = sa.id_servicio

UNION

SELECT
    h.id_huesped,
    h.nombre_completo,
    sa.nombre AS servicio,
    'Restaurante' AS tipo_servicio,
    tr.hora_entrada AS hora_inicio,
    tr.hora_salida AS hora_fin
FROM huesped h
JOIN reserva r ON h.id_huesped = r.id_huesped
JOIN ticket_restaurante tr ON r.id_reserva = tr.id_reserva
JOIN restaurante res ON res.id_servicio = (SELECT id_servicio FROM servicios_adicionales WHERE nombre ILIKE 'Restaurante' LIMIT 1)
JOIN servicios_adicionales sa ON res.id_servicio = sa.id_servicio;

-- Por cada uno de los huespedes
SELECT * FROM vista_servicios_contratados_por_huesped;
-- Por huésped específico
SELECT * FROM vista_servicios_contratados_por_huesped
WHERE id_huesped = 7;
-- Por tipo de servicio
SELECT * FROM vista_servicios_contratados_por_huesped
WHERE tipo_servicio = 'Gimnasio';

-- 4. Disparador (trigger)
CREATE OR REPLACE FUNCTION info_reserva_completa(p_id_reserva INTEGER)
RETURNS TABLE (
    nombre_completo VARCHAR,
    telefono VARCHAR,
    email VARCHAR,
    direccion TEXT,
    numero_habitacion INTEGER,
    piso INTEGER,
    tipo_habitacion VARCHAR,
    precio_por_noche DECIMAL,
    estado_habitacion VARCHAR,
    fecha_entrada DATE,
    fecha_salida DATE,
    numero_personas INTEGER,
    metodo_pago VARCHAR,
    estado_reserva VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        h.nombre_completo,
        h.telefono,
        h.email,
        h.direccion,
        ha.numero,
        ha.piso,
        ha.tipo,
        ha.precio_por_noche,
        ha.estado,
        r.fecha_entrada,
        r.fecha_salida,
        r.numero_personas,
        r.metodo_pago,
        r.estado
    FROM reserva r
    INNER JOIN huesped h ON r.id_huesped = h.id_huesped
    INNER JOIN habitacion ha ON r.id_habitacion = ha.id_habitacion
    WHERE r.id_reserva = p_id_reserva;
END;
$$ LANGUAGE plpgsql;
-- ejecucion
SELECT * FROM info_reserva_completa(5);

-- 5. Un catalogo en forma de vista que permita visualizar lo que se consumio por ticket ---------
CREATE OR REPLACE VIEW catalogo_consumo_restaurante AS
SELECT
    tr.id_ticket,
    tr.id_reserva,
    h.nombre_completo,
    m.nombre AS platillo,
    m.tipo,
    dt.cantidad,
    m.precio,
    (dt.cantidad * m.precio) AS total_parcial,
    tr.hora_entrada,
    tr.hora_salida
FROM ticket_restaurante tr
JOIN detalle_ticket dt ON tr.id_ticket = dt.id_ticket
JOIN menu m ON dt.id_platillo = m.id_platillo
JOIN reserva r ON tr.id_reserva = r.id_reserva
JOIN huesped h ON r.id_huesped = h.id_huesped
ORDER BY tr.id_ticket, m.nombre;

	SELECT * FROM catalogo_consumo_restaurante;
	-- Obtener el total por ticket
		SELECT
	    id_ticket,
	    id_reserva,
	    nombre_completo,
	    SUM(total_parcial) AS total_cuenta
		FROM catalogo_consumo_restaurante
		GROUP BY id_ticket, id_reserva, nombre_completo
		ORDER BY id_ticket;