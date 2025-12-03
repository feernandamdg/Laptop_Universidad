-- Base de Datos Hotel - Esquema Normalizado
-- Eliminación de tablas si existen (en orden correcto por dependencias)
DROP TABLE IF EXISTS detalle_ticket CASCADE;
DROP TABLE IF EXISTS ticket_restaurante CASCADE;
DROP TABLE IF EXISTS menu CASCADE;
DROP TABLE IF EXISTS uso_transporte CASCADE;
DROP TABLE IF EXISTS uso_gimnasio CASCADE;
DROP TABLE IF EXISTS uso_spa CASCADE;
DROP TABLE IF EXISTS tratamiento CASCADE;
DROP TABLE IF EXISTS restaurante CASCADE;
DROP TABLE IF EXISTS transporte CASCADE;
DROP TABLE IF EXISTS gimnasio CASCADE;
DROP TABLE IF EXISTS spa CASCADE;
DROP TABLE IF EXISTS servicios_adicionales CASCADE;
DROP TABLE IF EXISTS reserva CASCADE;
DROP TABLE IF EXISTS habitacion CASCADE;
DROP TABLE IF EXISTS huesped CASCADE;

-- Tabla Habitación
CREATE TABLE habitacion (
    id_habitacion SERIAL PRIMARY KEY,
    numero INTEGER NOT NULL UNIQUE,
    piso INTEGER NOT NULL,
    tipo VARCHAR(20) NOT NULL CHECK (tipo IN ('Individual', 'Doble', 'Suite')),
    precio_por_noche DECIMAL(10,2) NOT NULL CHECK (precio_por_noche > 0),
    estado VARCHAR(20) NOT NULL DEFAULT 'Disponible' CHECK (estado IN ('Disponible', 'Ocupada', 'En Mantenimiento'))
);

-- Tabla Huésped
CREATE TABLE huesped (
    id_huesped SERIAL PRIMARY KEY,
    nombre_completo VARCHAR(100) NOT NULL,
    telefono VARCHAR(20) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    direccion TEXT NOT NULL,
    fecha_registro DATE NOT NULL DEFAULT CURRENT_DATE
);

-- Tabla Reserva
CREATE TABLE reserva (
    id_reserva SERIAL PRIMARY KEY,
    fecha_entrada DATE NOT NULL,
    fecha_salida DATE NOT NULL,
    numero_personas INTEGER NOT NULL CHECK (numero_personas > 0),
    metodo_pago VARCHAR(30) NOT NULL CHECK (metodo_pago IN ('Efectivo', 'Tarjeta Credito', 'Tarjeta Debito', 'Transferencia')),
    estado VARCHAR(20) NOT NULL DEFAULT 'Activa' CHECK (estado IN ('Activa', 'Cancelada', 'Finalizada')),
    id_huesped INTEGER NOT NULL,
    id_habitacion INTEGER NOT NULL,
    CONSTRAINT fk_reserva_huesped FOREIGN KEY (id_huesped) REFERENCES huesped(id_huesped) ON DELETE CASCADE,
    CONSTRAINT fk_reserva_habitacion FOREIGN KEY (id_habitacion) REFERENCES habitacion(id_habitacion) ON DELETE RESTRICT,
    CONSTRAINT chk_fechas CHECK (fecha_salida > fecha_entrada)
);

-- Tabla Servicios Adicionales (Padre)
CREATE TABLE servicios_adicionales (
    id_servicio SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    hora_apertura TIME NOT NULL,
    hora_cierre TIME NOT NULL,
    CONSTRAINT chk_horario CHECK (hora_cierre > hora_apertura)
);

-- Tabla Spa (Herencia)
CREATE TABLE spa (
    id_servicio INTEGER PRIMARY KEY,
    capacidad INTEGER NOT NULL DEFAULT 50 CHECK (capacidad > 0),
    CONSTRAINT fk_spa_servicio FOREIGN KEY (id_servicio) REFERENCES servicios_adicionales(id_servicio) ON DELETE CASCADE
);

-- Tabla Tratamiento (Entidad débil de Spa)
CREATE TABLE tratamiento (
    id_tratamiento SERIAL PRIMARY KEY,
    id_servicio INTEGER NOT NULL,
    nombre VARCHAR(80) NOT NULL,
    descripcion TEXT,
    costo DECIMAL(10,2) NOT NULL CHECK (costo > 0),
    duracion_minutos INTEGER NOT NULL CHECK (duracion_minutos > 0),
    CONSTRAINT fk_tratamiento_spa FOREIGN KEY (id_servicio) REFERENCES spa(id_servicio) ON DELETE CASCADE
);

-- Tabla Uso Spa (Relación derivada)
CREATE TABLE uso_spa (
    id_uso_spa SERIAL PRIMARY KEY,
    id_reserva INTEGER NOT NULL,
    id_tratamiento INTEGER NOT NULL,
    hora_inicio TIMESTAMP NOT NULL,
    hora_fin TIMESTAMP NOT NULL,
    CONSTRAINT fk_uso_spa_reserva FOREIGN KEY (id_reserva) REFERENCES reserva(id_reserva) ON DELETE CASCADE,
    CONSTRAINT fk_uso_spa_tratamiento FOREIGN KEY (id_tratamiento) REFERENCES tratamiento(id_tratamiento) ON DELETE RESTRICT,
    CONSTRAINT chk_horario_spa CHECK (hora_fin > hora_inicio)
);

-- Tabla Gimnasio (Herencia)
CREATE TABLE gimnasio (
    id_servicio INTEGER PRIMARY KEY,
    capacidad INTEGER NOT NULL CHECK (capacidad > 0),
    CONSTRAINT fk_gimnasio_servicio FOREIGN KEY (id_servicio) REFERENCES servicios_adicionales(id_servicio) ON DELETE CASCADE
);

-- Tabla Uso Gimnasio (Relación derivada)
CREATE TABLE uso_gimnasio (
    id_registro SERIAL PRIMARY KEY,
    id_reserva INTEGER NOT NULL,
    hora_entrada TIMESTAMP NOT NULL,
    hora_salida TIMESTAMP NOT NULL,
    CONSTRAINT fk_uso_gimnasio_reserva FOREIGN KEY (id_reserva) REFERENCES reserva(id_reserva) ON DELETE CASCADE,
    CONSTRAINT chk_horario_gimnasio CHECK (hora_salida > hora_entrada)
);

-- Tabla Transporte (Herencia)
CREATE TABLE transporte (
    id_servicio INTEGER PRIMARY KEY,
    tipo_vehiculo VARCHAR(20) NOT NULL CHECK (tipo_vehiculo IN ('Sedan', 'Pick-up', 'Camioneta familiar')),
    tarifa_hora DECIMAL(10,2) NOT NULL CHECK (tarifa_hora > 0),
    capacidad INTEGER NOT NULL CHECK (capacidad > 0),
    CONSTRAINT fk_transporte_servicio FOREIGN KEY (id_servicio) REFERENCES servicios_adicionales(id_servicio) ON DELETE CASCADE
);

-- Tabla Uso Transporte (Relación derivada) - SIN costo_total
CREATE TABLE uso_transporte (
    id_registro SERIAL PRIMARY KEY,
    id_reserva INTEGER NOT NULL,
    fecha_hora_salida TIMESTAMP NOT NULL,
    fecha_hora_regreso TIMESTAMP NOT NULL,
    duracion_horas DECIMAL(4,2) NOT NULL CHECK (duracion_horas > 0),
    CONSTRAINT fk_uso_transporte_reserva FOREIGN KEY (id_reserva) REFERENCES reserva(id_reserva) ON DELETE CASCADE,
    CONSTRAINT chk_horario_transporte CHECK (fecha_hora_regreso > fecha_hora_salida)
);

-- Tabla Restaurante (Herencia)
CREATE TABLE restaurante (
    id_servicio INTEGER PRIMARY KEY,
    CONSTRAINT fk_restaurante_servicio FOREIGN KEY (id_servicio) REFERENCES servicios_adicionales(id_servicio) ON DELETE CASCADE
);

-- Tabla Menú (Entidad débil de Restaurante)
CREATE TABLE menu (
    id_platillo SERIAL PRIMARY KEY,
    id_servicio INTEGER NOT NULL,
    nombre VARCHAR(80) NOT NULL,
    descripcion TEXT,
    precio DECIMAL(10,2) NOT NULL CHECK (precio > 0),
    tipo VARCHAR(20) NOT NULL CHECK (tipo IN ('Entrada', 'Fuerte', 'Postre', 'Bebida')),
    CONSTRAINT fk_menu_restaurante FOREIGN KEY (id_servicio) REFERENCES restaurante(id_servicio) ON DELETE CASCADE
);

-- Tabla Ticket Restaurante (Relación derivada) - SIN total
CREATE TABLE ticket_restaurante (
    id_ticket SERIAL PRIMARY KEY,
    id_reserva INTEGER NOT NULL,
    hora_entrada TIMESTAMP NOT NULL,
    hora_salida TIMESTAMP,
    CONSTRAINT fk_ticket_reserva FOREIGN KEY (id_reserva) REFERENCES reserva(id_reserva) ON DELETE CASCADE,
    CONSTRAINT chk_horario_restaurante CHECK (hora_salida IS NULL OR hora_salida > hora_entrada)
);

-- Tabla Detalle Ticket (Entidad débil) - SIN subtotal
CREATE TABLE detalle_ticket (
    id_detalle SERIAL PRIMARY KEY,
    id_ticket INTEGER NOT NULL,
    id_platillo INTEGER NOT NULL,
    cantidad INTEGER NOT NULL CHECK (cantidad > 0),
    CONSTRAINT fk_detalle_ticket FOREIGN KEY (id_ticket) REFERENCES ticket_restaurante(id_ticket) ON DELETE CASCADE,
    CONSTRAINT fk_detalle_platillo FOREIGN KEY (id_platillo) REFERENCES menu(id_platillo) ON DELETE RESTRICT
);

-- Índices adicionales para mejorar rendimiento
CREATE INDEX idx_reserva_fechas ON reserva(fecha_entrada, fecha_salida);
CREATE INDEX idx_reserva_huesped ON reserva(id_huesped);
CREATE INDEX idx_reserva_habitacion ON reserva(id_habitacion);
CREATE INDEX idx_habitacion_estado ON habitacion(estado);
CREATE INDEX idx_habitacion_tipo ON habitacion(tipo);
CREATE INDEX idx_uso_spa_reserva ON uso_spa(id_reserva);
CREATE INDEX idx_uso_gimnasio_reserva ON uso_gimnasio(id_reserva);
CREATE INDEX idx_uso_transporte_reserva ON uso_transporte(id_reserva);
CREATE INDEX idx_ticket_reserva ON ticket_restaurante(id_reserva);
CREATE INDEX idx_detalle_ticket ON detalle_ticket(id_ticket);
CREATE INDEX idx_menu_tipo ON menu(tipo);

-- Comentarios en tablas
COMMENT ON TABLE habitacion IS 'Registro de habitaciones del hotel';
COMMENT ON TABLE huesped IS 'Registro de huéspedes del hotel';
COMMENT ON TABLE reserva IS 'Reservas realizadas por los huéspedes';
COMMENT ON TABLE servicios_adicionales IS 'Servicios adicionales del hotel (tabla padre)';
COMMENT ON TABLE spa IS 'Servicio de spa del hotel';
COMMENT ON TABLE tratamiento IS 'Tratamientos disponibles en el spa';
COMMENT ON TABLE uso_spa IS 'Registro de uso del spa por reserva';
COMMENT ON TABLE gimnasio IS 'Servicio de gimnasio del hotel';
COMMENT ON TABLE uso_gimnasio IS 'Registro de uso del gimnasio por reserva';
COMMENT ON TABLE transporte IS 'Servicio de transporte del hotel';
COMMENT ON TABLE uso_transporte IS 'Registro de uso del transporte por reserva';
COMMENT ON TABLE restaurante IS 'Servicio de restaurante del hotel';
COMMENT ON TABLE menu IS 'Platillos disponibles en el restaurante';
COMMENT ON TABLE ticket_restaurante IS 'Tickets de consumo en el restaurante';
COMMENT ON TABLE detalle_ticket IS 'Detalle de platillos por ticket';

-- Inserción de datos de ejemplo
-- Servicios Adicionales
INSERT INTO servicios_adicionales (nombre, hora_apertura, hora_cierre) VALUES
('Spa Relajante', '08:00', '20:00'),
('Gimnasio Premium', '06:00', '22:00'),
('Transporte Ejecutivo', '00:00', '23:59'),
('Restaurante Gourmet', '07:00', '23:00');

-- Herencia - Servicios específicos
INSERT INTO spa (id_servicio, capacidad) VALUES (1, 50); -- el costo se especifica en tratamiento
INSERT INTO gimnasio (id_servicio, capacidad) VALUES (2, 30);
INSERT INTO transporte (id_servicio, tipo_vehiculo, tarifa_hora, capacidad) 
VALUES (3, 'Sedan', 150.00, 4);
INSERT INTO restaurante (id_servicio) VALUES (4);

-- Habitaciones
INSERT INTO habitacion (numero, piso, tipo, precio_por_noche, estado) VALUES
(101, 1, 'Individual', 800.00, 'Disponible'),
(102, 1, 'Doble', 1200.00, 'Disponible'),
(103, 1, 'Individual', 800.00, 'Disponible'),
(104, 1, 'Doble', 1200.00, 'Disponible'),
(105, 1, 'Individual', 800.00, 'Disponible'),
(106, 1, 'Doble', 1200.00, 'Disponible'),
(107, 1, 'Individual', 800.00, 'Disponible'),
(108, 1, 'Doble', 1200.00, 'Disponible'),
(109, 1, 'Individual', 800.00, 'Disponible'),
(110, 1, 'Doble', 1200.00, 'Disponible'),
(201, 2, 'Individual', 800.00, 'Disponible'),
(202, 2, 'Doble', 1200.00, 'Disponible'),
(203, 2, 'Individual', 800.00, 'Disponible'),
(204, 2, 'Doble', 1200.00, 'Disponible'),
(205, 2, 'Individual', 800.00, 'Disponible'),
(206, 2, 'Doble', 1200.00, 'Disponible'),
(207, 2, 'Suite', 2500.00, 'Disponible'),
(208, 2, 'Doble', 1200.00, 'Ocupada'),
(209, 2, 'Individual', 800.00, 'Disponible'),
(210, 2, 'Doble', 1200.00, 'Disponible'),
(301, 3, 'Individual', 800.00, 'Disponible'),
(302, 3, 'Doble', 1200.00, 'Disponible'),
(303, 3, 'Individual', 800.00, 'Disponible'),
(304, 3, 'Doble', 1200.00, 'Disponible'),
(305, 3, 'Individual', 800.00, 'Disponible'),
(306, 3, 'Doble', 1200.00, 'Disponible'),
(307, 3, 'Individual', 800.00, 'Disponible'),
(308, 3, 'Doble', 1200.00, 'Disponible'),
(309, 3, 'Individual', 800.00, 'Disponible'),
(310, 3, 'Doble', 1200.00, 'Disponible'),
(401, 4, 'Individual', 800.00, 'Disponible'),
(402, 4, 'Doble', 1200.00, 'Disponible'),
(403, 4, 'Individual', 800.00, 'Disponible'),
(404, 4, 'Doble', 1200.00, 'Disponible'),
(405, 4, 'Individual', 800.00, 'Disponible'),
(406, 4, 'Doble', 1200.00, 'Disponible'),
(407, 4, 'Individual', 800.00, 'Disponible'),
(408, 4, 'Doble', 1200.00, 'Disponible'),
(409, 4, 'Individual', 800.00, 'Disponible'),
(410, 4, 'Doble', 1200.00, 'Disponible'),
(501, 5, 'Individual', 800.00, 'Disponible'),
(502, 5, 'Suite', 4000.00, 'Ocupada'),
(503, 5, 'Individual', 800.00, 'Disponible'),
(504, 5, 'Suite', 4000.00, 'Disponible'),
(505, 5, 'Individual', 800.00, 'En Mantenimiento'),
(506, 5, 'Suite', 4000.00, 'Disponible'),
(507, 5, 'Individual', 800.00, 'Disponible'),
(508, 5, 'Suite', 4000.00, 'Ocupada'),
(509, 5, 'Individual', 800.00, 'Disponible'),
(510, 5, 'Suite', 4000.00, 'Disponible');

-- Huéspedes
INSERT INTO huesped (nombre_completo, telefono, email, direccion) VALUES
('Juan Pérez García', '5551234567', 'juan.perez@email.com', 'Av. Reforma 123, CDMX'),
('María López Santos', '5559876543', 'maria.lopez@email.com', 'Calle Insurgentes 456, CDMX'),
('Majo Hidalgo Urrutia', '5088654327', 'majo.hidalgo@email.com', 'Cerrada Plan de San Luis 90, CDMX'),
('Emily Salazar García', '5559084567', 'emily.salazar@email.com', 'Av. Indistrial 193, GDL'),
('Irina Vaeva Portilla', '5559875433', 'irina.vaeva@email.com', 'Eulario Bonfil 180, COAH'),
('Cristiano Santos Margarito', '2808654327', 'cristiano.santos@email.com', 'Cerrada Plan de San Luis 90, CDMX'),
('Yanira Pachuca Herrera', '5098234997', 'yanira.pachuca@email.com', 'Calle Roma 99, VER'),
('María Mora Cleiton', '9999876543', 'maria.mora@email.com', 'Av. Cortéz 766, MEX'),
('Paige Bueckers Smith', '5088654327', 'paige.bueckers@email.com', 'Calle Matamoros 105, MICH'),
('Dijonai Carrington Miller', '5559084227', 'dijonai.carrington@email.com', 'Av. dallas 193, OAX'),
('Noemi Oliva Rivera', '0159875433', 'noemi.oliva@email.com', 'Calle Tamaulipas 33, TIJ'),
('Cristina Bautista Bautista', '2808657877', 'cristina.bautista@email.com', 'Calle Jalisco 99, NL'),
('Carlos Ramírez Díaz', '5553217890', 'carlos.ramirez@email.com', 'Calle Juárez 101, CDMX'),
('Laura Hernández Vega', '5556789012', 'laura.hernandez@email.com', 'Av. Chapultepec 230, CDMX'),
('Ana Torres Méndez', '5554321987', 'ana.torres@email.com', 'Calle Liverpool 56, CDMX'),
('Miguel Sánchez Ruiz', '5551092837', 'miguel.sanchez@email.com', 'Av. Universidad 400, CDMX'),
('Sofía Navarro Beltrán', '5558392019', 'sofia.navarro@email.com', 'Calz. Tlalpan 89, CDMX'),
('Ricardo Flores Gómez', '5552948376', 'ricardo.flores@email.com', 'Calle Colima 78, CDMX'),
('Paola Ríos Muñoz', '5557482910', 'paola.rios@email.com', 'Av. Coyoacán 512, CDMX'),
('José Antonio Morales', '5553829104', 'jose.morales@email.com', 'Calle Oaxaca 22, CDMX'),
('Verónica Castillo Pineda', '5559238475', 'veronica.castillo@email.com', 'Av. Revolución 330, CDMX'),
('Eduardo Jiménez Robles', '5551203948', 'eduardo.jimenez@email.com', 'Calle Niza 44, CDMX'),
('Daniela Estrada Núñez', '5553847290', 'daniela.estrada@email.com', 'Calle Sonora 8, CDMX'),
('Luis Ángel Cortés', '5559382710', 'luis.cortes@email.com', 'Av. Patriotismo 65, CDMX'),
('Alejandra Vargas Silva', '5557293048', 'alejandra.vargas@email.com', 'Calle Álvaro Obregón 150, CDMX'),
('Fernando Herrera Molina', '5552839471', 'fernando.herrera@email.com', 'Av. División del Norte 92, CDMX'),
('Andrea Mendoza Salas', '5558493021', 'andrea.mendoza@email.com', 'Calle Puebla 34, CDMX'),
('Iván Gómez Álvarez', '5553928401', 'ivan.gomez@email.com', 'Calle Tamaulipas 104, CDMX'),
('Gabriela Pacheco Ramos', '5557892043', 'gabriela.pacheco@email.com', 'Calle Tlaxcala 50, CDMX'),
('Santiago Ortega Fuentes', '5553827409', 'santiago.ortega@email.com', 'Calle Durango 95, CDMX'),
('Natalia Luna Carvajal', '5558924730', 'natalia.luna@email.com', 'Av. Cuauhtémoc 342, CDMX'),
('Jorge Barrera Solís', '5551203847', 'jorge.barrera@email.com', 'Calle Jalapa 67, CDMX'),
('Camila Reyes Domínguez', '5553049582', 'camila.reyes@email.com', 'Av. San Antonio 125, CDMX'),
('Emilio Castro Nieto', '5551820394', 'emilio.castro@email.com', 'Calle Campeche 66, CDMX'),
('Marina Delgado Ayala', '5557283910', 'marina.delgado@email.com', 'Av. Xola 209, CDMX'),
('Héctor Luna Vargas', '5553849201', 'hector.luna@email.com', 'Calle Medellín 198, CDMX'),
('Brenda Carrillo Torres', '5551823749', 'brenda.carrillo@email.com', 'Calle Baja California 15, CDMX'),
('Diego Salinas Rocha', '5552948173', 'diego.salinas@email.com', 'Av. Del Valle 91, CDMX'),
('Lucía Peña Robledo', '5558391042', 'lucia.pena@email.com', 'Calle Zacatecas 109, CDMX'),
('Alan Muñoz Rivera', '5553948201', 'alan.munoz@email.com', 'Calle Chiapas 40, CDMX'),
('Isabel Cordero Lara', '5551928374', 'isabel.cordero@email.com', 'Av. Benito Juárez 310, CDMX'),
('Tomás Aguilar Cabrera', '5559473820', 'tomas.aguilar@email.com', 'Calle Miguel Ángel 72, CDMX'),
('Claudia Serrano Méndez', '5553842017', 'claudia.serrano@email.com', 'Av. Copilco 200, CDMX'),
('Esteban Bravo Sánchez', '5558493720', 'esteban.bravo@email.com', 'Calle Popocatépetl 85, CDMX'),
('Renata Rivera León', '5552049387', 'renata.rivera@email.com', 'Calle Holbein 17, CDMX'),
('Ángel Navarro Zúñiga', '5557382049', 'angel.navarro@email.com', 'Av. Universidad 620, CDMX'),
('Montserrat Rubio Duarte', '5553947582', 'montserrat.rubio@email.com', 'Calle Arenal 31, CDMX'),
('Bruno Pineda Estrada', '5559283741', 'bruno.pineda@email.com', 'Calle Río Tíber 11, CDMX'),
('Valeria Lozano Cruz', '5558203947', 'valeria.lozano@email.com', 'Av. Santa María 56, CDMX'),
('Adrián Vega Palacios', '5553029481', 'adrian.vega@email.com', 'Calle Río Nazas 73, CDMX');

-- Reservas
INSERT INTO reserva (fecha_entrada, fecha_salida, numero_personas, metodo_pago, estado, id_huesped, id_habitacion) VALUES
('2025-07-01', '2025-07-05', 2, 'Tarjeta Credito', 'Activa', 1, 10),
('2025-07-10', '2025-07-15', 1, 'Efectivo', 'Finalizada', 2, 5),
('2025-07-03', '2025-07-07', 3, 'Transferencia', 'Activa', 3, 12),
('2025-07-08', '2025-07-12', 2, 'Tarjeta Debito', 'Cancelada', 4, 18),
('2025-07-05', '2025-07-10', 4, 'Efectivo', 'Activa', 5, 25),
('2025-07-15', '2025-07-18', 1, 'Tarjeta Credito', 'Finalizada', 6, 8),
('2025-07-20', '2025-07-25', 2, 'Transferencia', 'Activa', 7, 4),
('2025-07-12', '2025-07-16', 3, 'Tarjeta Debito', 'Activa', 8, 30),
('2025-07-17', '2025-07-21', 2, 'Efectivo', 'Activa', 9, 3),
('2025-07-22', '2025-07-28', 5, 'Tarjeta Credito', 'Cancelada', 10, 9),
('2025-07-04', '2025-07-08', 2, 'Transferencia', 'Finalizada', 11, 22),
('2025-07-09', '2025-07-11', 1, 'Tarjeta Debito', 'Activa', 12, 14),
('2025-07-06', '2025-07-10', 3, 'Efectivo', 'Activa', 13, 7),
('2025-07-13', '2025-07-16', 2, 'Transferencia', 'Finalizada', 14, 11),
('2025-07-18', '2025-07-22', 1, 'Tarjeta Credito', 'Activa', 15, 20),
('2025-07-24', '2025-07-27', 2, 'Tarjeta Debito', 'Cancelada', 16, 21),
('2025-07-26', '2025-07-31', 4, 'Transferencia', 'Activa', 17, 33),
('2025-07-07', '2025-07-12', 3, 'Tarjeta Credito', 'Finalizada', 18, 6),
('2025-07-15', '2025-07-17', 2, 'Tarjeta Debito', 'Activa', 19, 16),
('2025-07-19', '2025-07-23', 1, 'Efectivo', 'Activa', 20, 1),
('2025-07-02', '2025-07-06', 2, 'Transferencia', 'Cancelada', 21, 2),
('2025-07-11', '2025-07-14', 3, 'Tarjeta Debito', 'Finalizada', 22, 28),
('2025-07-16', '2025-07-20', 2, 'Tarjeta Credito', 'Activa', 23, 15),
('2025-07-21', '2025-07-26', 1, 'Efectivo', 'Activa', 24, 32),
('2025-07-29', '2025-08-03', 5, 'Transferencia', 'Finalizada', 25, 19),
('2025-07-01', '2025-07-04', 2, 'Tarjeta Debito', 'Activa', 26, 17),
('2025-07-05', '2025-07-09', 3, 'Tarjeta Credito', 'Cancelada', 27, 23),
('2025-07-08', '2025-07-12', 2, 'Efectivo', 'Activa', 28, 24),
('2025-07-10', '2025-07-13', 1, 'Transferencia', 'Finalizada', 29, 27),
('2025-07-14', '2025-07-18', 2, 'Tarjeta Debito', 'Activa', 30, 36),
('2025-07-20', '2025-07-23', 4, 'Efectivo', 'Cancelada', 31, 26),
('2025-07-22', '2025-07-26', 2, 'Tarjeta Credito', 'Activa', 32, 29),
('2025-07-25', '2025-07-30', 3, 'Transferencia', 'Finalizada', 33, 34),
('2025-07-03', '2025-07-07', 1, 'Tarjeta Debito', 'Activa', 34, 31),
('2025-07-09', '2025-07-12', 2, 'Tarjeta Credito', 'Activa', 35, 13),
('2025-07-13', '2025-07-15', 3, 'Efectivo', 'Finalizada', 36, 35),
('2025-07-17', '2025-07-20', 2, 'Transferencia', 'Cancelada', 37, 37),
('2025-07-19', '2025-07-24', 1, 'Tarjeta Debito', 'Activa', 38, 38),
('2025-07-26', '2025-07-30', 3, 'Efectivo', 'Finalizada', 39, 40),
('2025-07-06', '2025-07-11', 2, 'Tarjeta Credito', 'Activa', 40, 41),
('2025-07-16', '2025-07-21', 4, 'Transferencia', 'Activa', 41, 42),
('2025-07-23', '2025-07-27', 3, 'Efectivo', 'Cancelada', 42, 43),
('2025-07-28', '2025-08-01', 2, 'Tarjeta Debito', 'Activa', 43, 44),
('2025-07-12', '2025-07-16', 1, 'Tarjeta Credito', 'Finalizada', 44, 45),
('2025-07-15', '2025-07-19', 2, 'Transferencia', 'Activa', 45, 46),
('2025-07-20', '2025-07-24', 3, 'Efectivo', 'Activa', 46, 47),
('2025-07-22', '2025-07-27', 2, 'Tarjeta Debito', 'Cancelada', 47, 48),
('2025-07-25', '2025-07-29', 1, 'Tarjeta Credito', 'Activa', 48, 49),
('2025-07-28', '2025-08-02', 2, 'Transferencia', 'Finalizada', 49, 50),
('2025-07-29', '2025-08-03', 2, 'Efectivo', 'Activa', 50, 39);

-- Tratamientos
INSERT INTO tratamiento (id_servicio, nombre, descripcion, costo, duracion_minutos) VALUES
(1, 'Masaje Relajante', 'Masaje corporal completo', 800.00, 60),
(1, 'Facial Hidratante', 'Tratamiento facial revitalizante', 600.00, 45),
(1, 'Exfoliación Corporal', 'Eliminación de células muertas con sales minerales', 700.00, 50),
(1, 'Masaje con Piedras Calientes', 'Masaje terapéutico con piedras volcánicas calientes', 850.00, 70),
(1, 'Envoltura de Chocolate', 'Tratamiento corporal hidratante con cacao natural', 950.00, 60);

INSERT INTO menu (id_servicio, nombre, descripcion, precio, tipo) VALUES
(4, 'Sopa de Tortilla', 'Sopa tradicional con tiras crujientes de tortilla, queso y aguacate', 120.00, 'Entrada'),
(4, 'Ceviche de Camarón', 'Camarones marinados en jugo de limón con jitomate y cebolla', 160.00, 'Entrada'),
(4, 'Bruschettas', 'Pan tostado con tomate, albahaca y aceite de oliva', 110.00, 'Entrada'),
(4, 'Crema de Champiñones', 'Sopa cremosa de champiñones con un toque de ajo', 125.00, 'Entrada'),
(4, 'Tacos Dorados', 'Tortillas crujientes rellenas de pollo con crema y queso', 130.00, 'Entrada'),
(4, 'Pechuga en Salsa de Mango', 'Pechuga de pollo bañada en salsa dulce de mango', 280.00, 'Fuerte'),
(4, 'Salmón a la Parrilla', 'Filete de salmón con salsa de limón y hierbas', 370.00, 'Fuerte'),
(4, 'Pasta Alfredo', 'Pasta cremosa con queso parmesano y pollo', 250.00, 'Fuerte'),
(4, 'Lasagna Boloñesa', 'Capas de pasta, carne molida y salsa de tomate casera', 290.00, 'Fuerte'),
(4, 'Rib Eye Asado', 'Corte jugoso a la parrilla con papas al romero', 480.00, 'Fuerte'),
(4, 'Brownie con Helado', 'Brownie tibio con bola de helado de vainilla', 120.00, 'Postre'),
(4, 'Pastel de Tres Leches', 'Esponjoso pastel tradicional bañado en tres tipos de leche', 110.00, 'Postre'),
(4, 'Panqueque de Plátano', 'Panqueque casero con miel y plátano caramelizado', 130.00, 'Postre'),
(4, 'Tarta de Queso', 'Cheesecake con base crujiente y mermelada de frutas rojas', 140.00, 'Postre'),
(4, 'Gelatina de Mosaico', 'Colorida gelatina de leche y sabores frutales', 90.00, 'Postre'),
(4, 'Agua de Jamaica', 'Bebida refrescante de flor de jamaica natural', 45.00, 'Bebida'),
(4, 'Limonada Natural', 'Refrescante jugo de limón con agua mineral', 50.00, 'Bebida'),
(4, 'Café Americano', 'Café negro recién preparado', 40.00, 'Bebida'),
(4, 'Té de Manzanilla', 'Infusión relajante de flores de manzanilla', 35.00, 'Bebida'),
(4, 'Chocolate Caliente', 'Bebida caliente de chocolate con leche', 60.00, 'Bebida'),
(4, 'Sopa Azteca', 'Caldo con tortilla frita, aguacate, chile pasilla y crema', 130.00, 'Entrada'),
(4, 'Empanadas de Queso', 'Empanadas horneadas rellenas de queso manchego', 125.00, 'Entrada'),
(4, 'Ensalada de Betabel', 'Betabel asado con nueces, queso de cabra y vinagreta', 140.00, 'Entrada'),
(4, 'Croquetas de Jamón', 'Croquetas crujientes rellenas de jamón y bechamel', 115.00, 'Entrada'),
(4, 'Crema de Elote', 'Sopa cremosa con granos de elote dorado', 120.00, 'Entrada'),
(4, 'Pollo al Chipotle', 'Pechuga bañada en salsa cremosa de chipotle', 260.00, 'Fuerte'),
(4, 'Fajitas Mixtas', 'Tiras de carne, pollo y vegetales al sartén', 310.00, 'Fuerte'),
(4, 'Chiles Rellenos', 'Chiles poblanos rellenos de queso, capeados en huevo', 275.00, 'Fuerte'),
(4, 'Camarones al Mojo', 'Camarones salteados con ajo y mantequilla', 340.00, 'Fuerte'),
(4, 'Paella Mexicana', 'Arroz con mariscos, pollo y especias', 390.00, 'Fuerte'),
(4, 'Helado Artesanal', 'Helado de vainilla con trozos de fruta natural', 95.00, 'Postre'),
(4, 'Pay de Limón', 'Base de galleta con crema de limón y merengue', 100.00, 'Postre'),
(4, 'Crepas de Nutella', 'Delgadas crepas rellenas de Nutella y plátano', 135.00, 'Postre'),
(4, 'Flan Napolitano', 'Postre tradicional con caramelo', 110.00, 'Postre'),
(4, 'Copita de Tiramisú', 'Postre italiano con café y queso mascarpone', 125.00, 'Postre'),
(4, 'Agua de Horchata', 'Bebida fría de arroz con canela', 45.00, 'Bebida'),
(4, 'Refresco Natural', 'Bebida de fruta del día con agua mineral', 48.00, 'Bebida'),
(4, 'Café Expreso', 'Café concentrado servido en taza pequeña', 42.00, 'Bebida'),
(4, 'Té Verde', 'Infusión natural con antioxidantes', 38.00, 'Bebida'),
(4, 'Smoothie de Mango', 'Bebida espesa con mango natural y yogurt', 65.00, 'Bebida'),
(4, 'Sopa de Verduras', 'Caldo caliente con vegetales frescos', 110.00, 'Entrada'),
(4, 'Pan de Ajo', 'Pan horneado con mantequilla de ajo y perejil', 85.00, 'Entrada'),
(4, 'Gazpacho', 'Sopa fría de tomate con pepino y pimiento', 95.00, 'Entrada'),
(4, 'Mini Quesadillas', 'Tortillas de maíz rellenas de queso Oaxaca', 105.00, 'Entrada'),
(4, 'Rollitos Primavera', 'Rollos fritos rellenos de verduras y carne', 120.00, 'Entrada'),
(4, 'Milanesa de Res', 'Carne empanizada acompañada con puré de papa', 270.00, 'Fuerte'),
(4, 'Tinga de Pollo', 'Pollo desmenuzado en salsa chipotle y jitomate', 240.00, 'Fuerte'),
(4, 'Albondigas en Chipotle', 'Albóndigas caseras bañadas en salsa de chipotle', 255.00, 'Fuerte'),
(4, 'Arrachera Marinada', 'Corte de res marinado y asado al gusto', 400.00, 'Fuerte');

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