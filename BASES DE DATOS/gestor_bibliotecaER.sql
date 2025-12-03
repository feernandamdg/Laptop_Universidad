-- Tabla CodigoPostal (para normalizar información de códigos postales)
CREATE TABLE CodigoPostal (
    codigo_postal VARCHAR(10) PRIMARY KEY,
    ciudad VARCHAR(100),
    estado VARCHAR(100),
    pais VARCHAR(100) DEFAULT 'México'
);

-- Tabla Direccion (normalizada para evitar redundancias)
CREATE TABLE Direccion (
    id_direccion SERIAL PRIMARY KEY,
    calle VARCHAR(150) NOT NULL,
    numero_exterior VARCHAR(20) NOT NULL,
    numero_interior VARCHAR(20),
    colonia VARCHAR(100) NOT NULL,
    codigo_postal VARCHAR(10) NOT NULL,
    FOREIGN KEY (codigo_postal) REFERENCES CodigoPostal(codigo_postal) ON UPDATE CASCADE
);

-- Tabla Usuario (normalizada con nombres desglosados)
CREATE TABLE Usuario (
    id_usuario SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    apellido_paterno VARCHAR(50) NOT NULL,
    apellido_materno VARCHAR(50),
    email VARCHAR(100) UNIQUE NOT NULL,
    telefono VARCHAR(20),
    id_direccion INTEGER NOT NULL,
    fecha_registro DATE NOT NULL DEFAULT CURRENT_DATE,
    estado VARCHAR(20) NOT NULL DEFAULT 'Activo' CHECK (estado IN ('Activo', 'Inactivo', 'Suspendido')),
    FOREIGN KEY (id_direccion) REFERENCES Direccion(id_direccion) ON DELETE RESTRICT
);

-- Tabla Autor
CREATE TABLE Autor (
    id_autor SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    apellido_paterno VARCHAR(50) NOT NULL,
    apellido_materno VARCHAR(50),
    nacionalidad VARCHAR(50),
    fecha_nacimiento DATE
);

-- Tabla Libro (normalizada con la información específica del libro)
CREATE TABLE Libro (
    id_libro SERIAL PRIMARY KEY,
    isbn VARCHAR(20) UNIQUE NOT NULL,
    titulo VARCHAR(200) NOT NULL,
    subtitulo VARCHAR(200),
    editorial VARCHAR(100),
    anio_publicacion INTEGER,
    edicion VARCHAR(50),
    estado VARCHAR(20) NOT NULL DEFAULT 'Disponible' CHECK (estado IN ('Disponible', 'Reservado', 'Prestado', 'En reparación', 'Extraviado'))
);

-- Tabla Libro_Autor (tabla de relación muchos a muchos entre Libro y Autor)
CREATE TABLE Libro_Autor (
    id_libro INTEGER,
    id_autor INTEGER,
    tipo_contribucion VARCHAR(50) DEFAULT 'Autor' CHECK (tipo_contribucion IN ('Autor', 'Co-autor', 'Editor', 'Traductor', 'Ilustrador')),
    PRIMARY KEY (id_libro, id_autor, tipo_contribucion),
    FOREIGN KEY (id_libro) REFERENCES Libro(id_libro) ON DELETE CASCADE,
    FOREIGN KEY (id_autor) REFERENCES Autor(id_autor) ON DELETE RESTRICT
);

-- Tabla Reserva (contiene la información de reservas de libros)
CREATE TABLE Reserva (
    id_reserva SERIAL PRIMARY KEY,
    id_usuario INTEGER NOT NULL,
    id_libro INTEGER NOT NULL,
    fecha_reserva TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    fecha_limite_retiro TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP + INTERVAL '3 days'),
    estado VARCHAR(20) NOT NULL DEFAULT 'Activa' CHECK (estado IN ('Activa', 'Finalizada', 'Cancelada', 'Vencida')),
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario) ON DELETE RESTRICT,
    FOREIGN KEY (id_libro) REFERENCES Libro(id_libro) ON DELETE RESTRICT
);

-- Restricciones adicionales

-- Un libro solo puede ser reservado por un usuario a la vez
CREATE OR REPLACE FUNCTION verificar_libro_disponible()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM Reserva 
        WHERE id_libro = NEW.id_libro 
        AND estado = 'Activa'
        AND id_reserva != COALESCE(NEW.id_reserva, 0)
    ) THEN
        RAISE EXCEPTION 'El libro ya está reservado por otro usuario';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER libro_disponible_reserva
BEFORE INSERT OR UPDATE ON Reserva
FOR EACH ROW
EXECUTE FUNCTION verificar_libro_disponible();

-- Actualizar estado del libro al reservarlo
CREATE OR REPLACE FUNCTION actualizar_estado_libro()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.estado = 'Activa' THEN
        UPDATE Libro SET estado = 'Reservado' WHERE id_libro = NEW.id_libro;
    ELSIF OLD.estado = 'Activa' AND (NEW.estado = 'Finalizada' OR NEW.estado = 'Cancelada' OR NEW.estado = 'Vencida') THEN
        UPDATE Libro SET estado = 'Disponible' WHERE id_libro = NEW.id_libro;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER actualizar_libro_al_reservar
AFTER INSERT OR UPDATE OF estado ON Reserva
FOR EACH ROW
EXECUTE FUNCTION actualizar_estado_libro();

-- Índices para mejorar el rendimiento
CREATE INDEX idx_libro_isbn ON Libro(isbn);
CREATE INDEX idx_usuario_nombres ON Usuario(apellido_paterno, apellido_materno, nombre);
CREATE INDEX idx_reserva_usuario ON Reserva(id_usuario);
CREATE INDEX idx_reserva_libro ON Reserva(id_libro);
CREATE INDEX idx_reserva_fecha ON Reserva(fecha_reserva);
CREATE INDEX idx_reserva_estado ON Reserva(estado);

-- Inserción de datos de ejemplo

-- Códigos postales
INSERT INTO CodigoPostal (codigo_postal, ciudad, estado) VALUES
('03100', 'Ciudad de México', 'CDMX'),
('03200', 'Ciudad de México', 'CDMX'),
('44100', 'Guadalajara', 'Jalisco'),
('64000', 'Monterrey', 'Nuevo León'),
('45050', 'Zapopan', 'Jalisco');

-- Direcciones
INSERT INTO Direccion (calle, numero_exterior, numero_interior, colonia, codigo_postal) VALUES
('Av. Universidad', '3000', NULL, 'Copilco Universidad', '03100'),
('Calle Manzanas', '123', '5', 'La Frutería', '03200'),
('Paseo de la Reforma', '222', NULL, 'Cuauhtémoc', '03100'),
('Avenida Chapultepec', '501', '3A', 'Americana', '44100'),
('Calzada Independencia', '788', NULL, 'Centro', '44100');

-- Usuarios
INSERT INTO Usuario (nombre, apellido_paterno, apellido_materno, email, telefono, id_direccion) VALUES
('Juan', 'García', 'Pérez', 'juan.garcia@email.com', '555-1234', 1),
('María', 'López', 'Sánchez', 'maria.lopez@email.com', '555-2345', 2),
('Carlos', 'Martínez', 'Rodríguez', 'carlos.martinez@email.com', '555-3456', 3),
('Ana', 'González', 'Fernández', 'ana.gonzalez@email.com', '555-4567', 4),
('Luis', 'Hernández', 'Torres', 'luis.hernandez@email.com', '555-5678', 5);

-- Autores
INSERT INTO Autor (nombre, apellido_paterno, apellido_materno, nacionalidad, fecha_nacimiento) VALUES
('Gabriel', 'García', 'Márquez', 'Colombiana', '1927-03-06'),
('Isabel', 'Allende', '', 'Chilena', '1942-08-02'),
('Octavio', 'Paz', '', 'Mexicana', '1914-03-31'),
('Jorge', 'Luis', 'Borges', 'Argentina', '1899-08-24'),
('Julio', 'Cortázar', '', 'Argentina', '1914-08-26'),
('Mario', 'Vargas', 'Llosa', 'Peruana', '1936-03-28'),
('Elena', 'Poniatowska', '', 'Mexicana', '1932-05-19');

-- Libros
INSERT INTO Libro (isbn, titulo, subtitulo, editorial, anio_publicacion, edicion, estado) VALUES
('9780307474728', 'Cien años de soledad', NULL, 'Vintage Español', 1967, 'Primera', 'Disponible'),
('9788401352898', 'La casa de los espíritus', NULL, 'Plaza & Janés', 1982, 'Segunda', 'Disponible'),
('9786073128827', 'El laberinto de la soledad', NULL, 'Fondo de Cultura Económica', 1950, 'Décima', 'Disponible'),
('9788499089508', 'Ficciones', NULL, 'Debolsillo', 1944, 'Quinta', 'Disponible'),
('9788420406794', 'Rayuela', NULL, 'Alfaguara', 1963, 'Tercera', 'Disponible'),
('9788490625583', 'La ciudad y los perros', NULL, 'Alfaguara', 1963, 'Segunda', 'Disponible'),
('9786073118477', 'La noche de Tlatelolco', 'Testimonios de historia oral', 'Era', 1971, 'Cuarta', 'Disponible');

-- Relación Libro-Autor
INSERT INTO Libro_Autor (id_libro, id_autor, tipo_contribucion) VALUES
(1, 1, 'Autor'),
(2, 2, 'Autor'),
(3, 3, 'Autor'),
(4, 4, 'Autor'),
(5, 5, 'Autor'),
(6, 6, 'Autor'),
(7, 7, 'Autor');

-- Reservas
INSERT INTO Reserva (id_usuario, id_libro, fecha_reserva, fecha_limite_retiro, estado) VALUES
(1, 1, CURRENT_TIMESTAMP - INTERVAL '5 days', CURRENT_TIMESTAMP - INTERVAL '2 days', 'Finalizada'),
(2, 3, CURRENT_TIMESTAMP - INTERVAL '3 days', CURRENT_TIMESTAMP, 'Activa'),
(3, 5, CURRENT_TIMESTAMP - INTERVAL '7 days', CURRENT_TIMESTAMP - INTERVAL '4 days', 'Vencida'),
(4, 2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP + INTERVAL '3 days', 'Activa'),
(5, 6, CURRENT_TIMESTAMP - INTERVAL '1 day', CURRENT_TIMESTAMP + INTERVAL '2 days', 'Activa');

-- Vistas

-- Vista de libros disponibles
CREATE OR REPLACE VIEW v_libros_disponibles AS
SELECT 
    l.id_libro,
    l.isbn,
    l.titulo,
    l.subtitulo,
    l.editorial,
    l.anio_publicacion,
    string_agg(a.nombre || ' ' || a.apellido_paterno || ' ' || COALESCE(a.apellido_materno, ''), ', ') AS autores
FROM 
    Libro l
LEFT JOIN 
    Libro_Autor la ON l.id_libro = la.id_libro
LEFT JOIN 
    Autor a ON la.id_autor = a.id_autor
WHERE 
    l.estado = 'Disponible'
GROUP BY 
    l.id_libro, l.isbn, l.titulo, l.subtitulo, l.editorial, l.anio_publicacion;

-- Vista de reservas activas
CREATE OR REPLACE VIEW v_reservas_activas AS
SELECT 
    r.id_reserva,
    u.id_usuario,
    u.nombre || ' ' || u.apellido_paterno || ' ' || COALESCE(u.apellido_materno, '') AS nombre_usuario,
    l.id_libro,
    l.isbn,
    l.titulo,
    r.fecha_reserva,
    r.fecha_limite_retiro,
    CASE
        WHEN r.fecha_limite_retiro < CURRENT_TIMESTAMP THEN 'Por vencer'
        ELSE 'Vigente'
    END AS estado_limite
FROM 
    Reserva r
JOIN 
    Usuario u ON r.id_usuario = u.id_usuario
JOIN 
    Libro l ON r.id_libro = l.id_libro
WHERE 
    r.estado = 'Activa';

-- Funciones

-- Función para realizar una reserva
CREATE OR REPLACE FUNCTION realizar_reserva(
    p_id_usuario INTEGER,
    p_id_libro INTEGER,
    p_dias_limite INTEGER DEFAULT 3
)
RETURNS INTEGER AS $$
DECLARE
    v_id_reserva INTEGER;
BEGIN
    -- Verificar si el libro está disponible
    IF NOT EXISTS (SELECT 1 FROM Libro WHERE id_libro = p_id_libro AND estado = 'Disponible') THEN
        RAISE EXCEPTION 'El libro no está disponible para reserva';
    END IF;
    
    -- Crear la reserva
    INSERT INTO Reserva (
        id_usuario,
        id_libro,
        fecha_reserva,
        fecha_limite_retiro,
        estado
    ) VALUES (
        p_id_usuario,
        p_id_libro,
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP + (p_dias_limite || ' days')::INTERVAL,
        'Activa'
    ) RETURNING id_reserva INTO v_id_reserva;
    
    RETURN v_id_reserva;
EXCEPTION
    WHEN OTHERS THEN
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Función para cancelar una reserva
CREATE OR REPLACE FUNCTION cancelar_reserva(
    p_id_reserva INTEGER
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Verificar si la reserva existe y está activa
    IF NOT EXISTS (SELECT 1 FROM Reserva WHERE id_reserva = p_id_reserva AND estado = 'Activa') THEN
        RAISE EXCEPTION 'La reserva no existe o no está activa';
    END IF;
    
    -- Actualizar la reserva a cancelada
    UPDATE Reserva SET estado = 'Cancelada' WHERE id_reserva = p_id_reserva;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Función para buscar libros por título o autor
CREATE OR REPLACE FUNCTION buscar_libros(
    p_termino VARCHAR
)
RETURNS TABLE (
    id_libro INTEGER,
    isbn VARCHAR,
    titulo VARCHAR,
    subtitulo VARCHAR,
    editorial VARCHAR,
    anio_publicacion INTEGER,
    autores TEXT,
    estado VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        l.id_libro,
        l.isbn,
        l.titulo,
        l.subtitulo,
        l.editorial,
        l.anio_publicacion,
        string_agg(a.nombre || ' ' || a.apellido_paterno || ' ' || COALESCE(a.apellido_materno, ''), ', ') AS autores,
        l.estado
    FROM 
        Libro l
    LEFT JOIN 
        Libro_Autor la ON l.id_libro = la.id_libro
    LEFT JOIN 
        Autor a ON la.id_autor = a.id_autor
    WHERE 
        LOWER(l.titulo) LIKE LOWER('%' || p_termino || '%')
        OR LOWER(l.subtitulo) LIKE LOWER('%' || p_termino || '%')
        OR LOWER(l.isbn) LIKE LOWER('%' || p_termino || '%')
        OR EXISTS (
            SELECT 1 FROM Autor a2
            JOIN Libro_Autor la2 ON a2.id_autor = la2.id_autor
            WHERE la2.id_libro = l.id_libro
            AND (
                LOWER(a2.nombre) LIKE LOWER('%' || p_termino || '%')
                OR LOWER(a2.apellido_paterno) LIKE LOWER('%' || p_termino || '%')
                OR LOWER(a2.apellido_materno) LIKE LOWER('%' || p_termino || '%')
            )
        )
    GROUP BY 
        l.id_libro, l.isbn, l.titulo, l.subtitulo, l.editorial, l.anio_publicacion, l.estado;
END;
$$ LANGUAGE plpgsql;

-- Procedimiento para actualizar estados de reservas vencidas
CREATE OR REPLACE PROCEDURE actualizar_reservas_vencidas()
LANGUAGE plpgsql AS $$
BEGIN
    -- Actualizar reservas vencidas
    UPDATE Reserva
    SET estado = 'Vencida'
    WHERE estado = 'Activa'
    AND fecha_limite_retiro < CURRENT_TIMESTAMP;
    
    -- Liberar libros de reservas vencidas
    UPDATE Libro l
    SET estado = 'Disponible'
    WHERE l.estado = 'Reservado'
    AND EXISTS (
        SELECT 1 FROM Reserva r
        WHERE r.id_libro = l.id_libro
        AND r.estado = 'Vencida'
    );
    
    COMMIT;
END;
$$;