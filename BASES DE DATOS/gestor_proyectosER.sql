-- FMV versión 2
-- Sistema de gestión de proyectos
-- Normalizado a 3FN

-- Eliminar tablas si existen para evitar errores al ejecutar varias veces
DROP TABLE IF EXISTS TareaEmpleado;
DROP TABLE IF EXISTS Tarea;
DROP TABLE IF EXISTS Proyecto;
DROP TABLE IF EXISTS Cliente;
DROP TABLE IF EXISTS Empleado;
DROP TABLE IF EXISTS Departamento;

-- Tabla Departamento
CREATE TABLE Departamento (
    id_departamento SERIAL PRIMARY KEY,
    codigo VARCHAR(10) NOT NULL UNIQUE, -- Código único del departamento
    nombre VARCHAR(100) NOT NULL,
    id_empleado_director INTEGER -- Se actualizará posteriormente con FK
);

-- Tabla Empleado
CREATE TABLE Empleado (
    id_empleado SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    primer_apellido VARCHAR(50) NOT NULL,
    segundo_apellido VARCHAR(50),
    cargo VARCHAR(100) NOT NULL,
    fecha_registro DATE NOT NULL DEFAULT CURRENT_DATE,
    id_departamento INTEGER NOT NULL,
    FOREIGN KEY (id_departamento) REFERENCES Departamento(id_departamento)
);

-- Actualizar la tabla Departamento para agregar la referencia al director
ALTER TABLE Departamento
    ADD CONSTRAINT fk_director
    FOREIGN KEY (id_empleado_director) REFERENCES Empleado(id_empleado);

-- Tabla Cliente
CREATE TABLE Cliente (
    id_cliente SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    direccion VARCHAR(200),
    email VARCHAR(100)
);

-- Tabla de teléfonos de cliente (para cumplir con 3FN)
CREATE TABLE TelefonoCliente (
    id_telefono SERIAL PRIMARY KEY,
    id_cliente INTEGER NOT NULL,
    telefono VARCHAR(20) NOT NULL,
    tipo VARCHAR(20), -- Móvil, Oficina, etc.
    FOREIGN KEY (id_cliente) REFERENCES Cliente(id_cliente) ON DELETE CASCADE
);

-- Tabla Proyecto
CREATE TABLE Proyecto (
    id_proyecto SERIAL PRIMARY KEY,
    codigo VARCHAR(20) NOT NULL UNIQUE,
    nombre VARCHAR(100) NOT NULL,
    fecha_inicio DATE NOT NULL,
    fecha_fin DATE,
    id_cliente INTEGER NOT NULL,
    FOREIGN KEY (id_cliente) REFERENCES Cliente(id_cliente),
    CHECK (fecha_fin IS NULL OR fecha_fin > fecha_inicio)
);

-- Tabla Tarea
CREATE TABLE Tarea (
    id_tarea SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    estado VARCHAR(20) NOT NULL CHECK (estado IN ('Pendiente', 'En progreso', 'Completada', 'Cancelada')),
    fecha_creacion DATE NOT NULL DEFAULT CURRENT_DATE,
    fecha_entrega DATE,
    recursos TEXT,
    id_proyecto INTEGER NOT NULL,
    FOREIGN KEY (id_proyecto) REFERENCES Proyecto(id_proyecto) ON DELETE CASCADE,
    CHECK (fecha_entrega IS NULL OR fecha_entrega >= fecha_creacion)
);

-- Tabla de relación muchos a muchos entre Tarea y Empleado
CREATE TABLE TareaEmpleado (
    id_tarea INTEGER NOT NULL,
    id_empleado INTEGER NOT NULL,
    fecha_asignacion DATE NOT NULL DEFAULT CURRENT_DATE,
    PRIMARY KEY (id_tarea, id_empleado),
    FOREIGN KEY (id_tarea) REFERENCES Tarea(id_tarea) ON DELETE CASCADE,
    FOREIGN KEY (id_empleado) REFERENCES Empleado(id_empleado) ON DELETE CASCADE
);

-- Índices para mejorar el rendimiento
CREATE INDEX idx_empleado_departamento ON Empleado(id_departamento);
CREATE INDEX idx_proyecto_cliente ON Proyecto(id_cliente);
CREATE INDEX idx_tarea_proyecto ON Tarea(id_proyecto);
CREATE INDEX idx_tarea_empleado_tarea ON TareaEmpleado(id_tarea);
CREATE INDEX idx_tarea_empleado_empleado ON TareaEmpleado(id_empleado);

-- Trigger para generar automáticamente el código del proyecto
CREATE OR REPLACE FUNCTION generar_codigo_proyecto()
RETURNS TRIGGER AS $$
BEGIN
    NEW.codigo := 'PROJ-' || TO_CHAR(CURRENT_DATE, 'YYYYMMDD') || '-' || LPAD(NEW.id_proyecto::TEXT, 4, '0');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_generar_codigo_proyecto
BEFORE INSERT ON Proyecto
FOR EACH ROW
EXECUTE FUNCTION generar_codigo_proyecto();

-- Función para verificar si un empleado es director de departamento
CREATE OR REPLACE FUNCTION es_director(p_id_empleado INTEGER)
RETURNS BOOLEAN AS $$
DECLARE
    v_es_director BOOLEAN;
BEGIN
    SELECT EXISTS(SELECT 1 FROM Departamento WHERE id_empleado_director = p_id_empleado) INTO v_es_director;
    RETURN v_es_director;
END;
$$ LANGUAGE plpgsql;

-- Ejemplo de consulta para obtener proyectos con sus tareas y empleados asignados
COMMENT ON FUNCTION es_director(INTEGER) IS 'Ejemplo de vista útil para el sistema:

CREATE VIEW vista_proyectos_completa AS
SELECT 
    p.id_proyecto, 
    p.codigo AS codigo_proyecto, 
    p.nombre AS nombre_proyecto,
    p.fecha_inicio, 
    p.fecha_fin,
    c.nombre AS nombre_cliente,
    t.id_tarea,
    t.nombre AS nombre_tarea,
    t.estado AS estado_tarea,
    t.fecha_entrega,
    e.id_empleado,
    e.nombre || '' '' || e.primer_apellido || COALESCE('' '' || e.segundo_apellido, '''') AS nombre_completo_empleado,
    e.cargo,
    d.nombre AS departamento,
    te.fecha_asignacion
FROM 
    Proyecto p
    JOIN Cliente c ON p.id_cliente = c.id_cliente
    LEFT JOIN Tarea t ON t.id_proyecto = p.id_proyecto
    LEFT JOIN TareaEmpleado te ON te.id_tarea = t.id_tarea
    LEFT JOIN Empleado e ON e.id_empleado = te.id_empleado
    LEFT JOIN Departamento d ON e.id_departamento = d.id_departamento
ORDER BY 
    p.id_proyecto, t.id_tarea, e.id_empleado;';