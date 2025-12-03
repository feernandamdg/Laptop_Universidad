-- Creación de tablas principales

-- Tabla Departamento
CREATE TABLE Departamento (
    id_departamento SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    id_director INTEGER UNIQUE -- Se establecerá como FK después de crear la tabla Empleado
);

-- Tabla Proyecto
CREATE TABLE Proyecto (
    id_proyecto SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    fecha_inicio DATE NOT NULL,
    fecha_fin DATE,
    estado VARCHAR(50) CHECK (estado IN ('Planificación', 'En progreso', 'Completado', 'Cancelado')),
    presupuesto DECIMAL(12, 2)
);

-- Tabla Empleado (entidad base)
CREATE TABLE Empleado (
    id_empleado SERIAL PRIMARY KEY,
    nombre_completo VARCHAR(150) NOT NULL,
    rol_general VARCHAR(50) NOT NULL CHECK (rol_general IN ('Programador', 'Analista', 'Tester', 'Director', 'Administrativo')),
    email VARCHAR(100) UNIQUE,
    telefono VARCHAR(20),
    fecha_contratacion DATE NOT NULL,
    id_departamento INTEGER,
    FOREIGN KEY (id_departamento) REFERENCES Departamento(id_departamento) ON DELETE SET NULL
);

-- Actualización de la FK de Director en Departamento
ALTER TABLE Departamento
ADD CONSTRAINT fk_departamento_director FOREIGN KEY (id_director) REFERENCES Empleado(id_empleado) ON DELETE SET NULL;

-- Tabla Empleado_Proyecto (relación muchos a muchos)
CREATE TABLE Empleado_Proyecto (
    id_empleado INTEGER,
    id_proyecto INTEGER,
    fecha_asignacion DATE NOT NULL DEFAULT CURRENT_DATE,
    rol_en_proyecto VARCHAR(100),
    PRIMARY KEY (id_empleado, id_proyecto),
    FOREIGN KEY (id_empleado) REFERENCES Empleado(id_empleado) ON DELETE CASCADE,
    FOREIGN KEY (id_proyecto) REFERENCES Proyecto(id_proyecto) ON DELETE CASCADE
);

-- Tablas para los tipos específicos de empleados (especialización)

-- Tabla Programador
CREATE TABLE Programador (
    id_empleado INTEGER PRIMARY KEY,
    lenguaje_dominante VARCHAR(50) NOT NULL,
    años_experiencia INTEGER,
    nivel VARCHAR(30) CHECK (nivel IN ('Junior', 'Semi-senior', 'Senior', 'Tech Lead')),
    FOREIGN KEY (id_empleado) REFERENCES Empleado(id_empleado) ON DELETE CASCADE
);

-- Tabla Analista
CREATE TABLE Analista (
    id_empleado INTEGER PRIMARY KEY,
    especialidad VARCHAR(100),
    años_experiencia INTEGER,
    nivel VARCHAR(30) CHECK (nivel IN ('Junior', 'Semi-senior', 'Senior', 'Lead')),
    FOREIGN KEY (id_empleado) REFERENCES Empleado(id_empleado) ON DELETE CASCADE
);

-- Tabla Tester
CREATE TABLE Tester (
    id_empleado INTEGER PRIMARY KEY,
    tipo_testing VARCHAR(50) CHECK (tipo_testing IN ('Manual', 'Automatizado', 'Ambos')),
    certificaciones_qa BOOLEAN DEFAULT FALSE,
    años_experiencia INTEGER,
    FOREIGN KEY (id_empleado) REFERENCES Empleado(id_empleado) ON DELETE CASCADE
);

-- Tablas relacionadas con Programador

-- Tabla Tarea
CREATE TABLE Tarea (
    id_tarea SERIAL PRIMARY KEY,
    descripcion TEXT NOT NULL,
    estado VARCHAR(30) CHECK (estado IN ('Por hacer', 'En progreso', 'En revisión', 'Completada')),
    fecha_asignacion DATE NOT NULL,
    fecha_vencimiento DATE,
    id_programador INTEGER,
    id_proyecto INTEGER NOT NULL,
    FOREIGN KEY (id_programador) REFERENCES Programador(id_empleado) ON DELETE SET NULL,
    FOREIGN KEY (id_proyecto) REFERENCES Proyecto(id_proyecto) ON DELETE CASCADE
);

-- Tablas relacionadas con Analista

-- Tabla HerramientaAnalisis
CREATE TABLE HerramientaAnalisis (
    id_herramienta SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    tipo VARCHAR(50),
    descripcion TEXT
);

-- Tabla Analista_Herramienta (relación muchos a muchos)
CREATE TABLE Analista_Herramienta (
    id_analista INTEGER,
    id_herramienta INTEGER,
    nivel_dominio VARCHAR(30) CHECK (nivel_dominio IN ('Básico', 'Intermedio', 'Avanzado', 'Experto')),
    PRIMARY KEY (id_analista, id_herramienta),
    FOREIGN KEY (id_analista) REFERENCES Analista(id_empleado) ON DELETE CASCADE,
    FOREIGN KEY (id_herramienta) REFERENCES HerramientaAnalisis(id_herramienta) ON DELETE CASCADE
);

-- Tabla Certificacion
CREATE TABLE Certificacion (
    id_certificacion SERIAL PRIMARY KEY,
    id_analista INTEGER NOT NULL,
    nombre VARCHAR(100) NOT NULL,
    entidad_emisora VARCHAR(100) NOT NULL,
    fecha_obtencion DATE NOT NULL,
    fecha_vencimiento DATE,
    FOREIGN KEY (id_analista) REFERENCES Analista(id_empleado) ON DELETE CASCADE
);

-- Tablas para clientes (jerarquía)

-- Tabla Cliente (entidad base)
CREATE TABLE Cliente (
    id_cliente SERIAL PRIMARY KEY,
    tipo VARCHAR(20) NOT NULL CHECK (tipo IN ('Particular', 'Empresa')),
    nombre VARCHAR(150) NOT NULL,
    email VARCHAR(100),
    telefono VARCHAR(20),
    fecha_registro DATE NOT NULL DEFAULT CURRENT_DATE
);

-- Tabla Particular (especialización de Cliente)
CREATE TABLE Particular (
    id_cliente INTEGER PRIMARY KEY,
    apellidos VARCHAR(100),
    documento_identidad VARCHAR(20) UNIQUE,
    fecha_nacimiento DATE,
    FOREIGN KEY (id_cliente) REFERENCES Cliente(id_cliente) ON DELETE CASCADE
);

-- Tabla Empresa (especialización de Cliente)
CREATE TABLE Empresa (
    id_cliente INTEGER PRIMARY KEY,
    razon_social VARCHAR(200) NOT NULL,
    nif VARCHAR(20) UNIQUE NOT NULL,
    sector VARCHAR(100),
    contacto_principal VARCHAR(100),
    FOREIGN KEY (id_cliente) REFERENCES Cliente(id_cliente) ON DELETE CASCADE
);

-- Tabla Reunion
CREATE TABLE Reunion (
    id_reunion SERIAL PRIMARY KEY,
    fecha_hora TIMESTAMP NOT NULL,
    duracion INTEGER, -- en minutos
    ubicacion VARCHAR(200),
    tema VARCHAR(200) NOT NULL,
    id_analista INTEGER,
    id_cliente INTEGER,
    id_proyecto INTEGER,
    FOREIGN KEY (id_analista) REFERENCES Analista(id_empleado) ON DELETE SET NULL,
    FOREIGN KEY (id_cliente) REFERENCES Cliente(id_cliente) ON DELETE SET NULL,
    FOREIGN KEY (id_proyecto) REFERENCES Proyecto(id_proyecto) ON DELETE CASCADE
);

-- Tabla ReportesReunion
CREATE TABLE ReportesReunion (
    id_reporte SERIAL PRIMARY KEY,
    id_reunion INTEGER NOT NULL,
    tipo_reporte VARCHAR(50) NOT NULL,
    contenido TEXT,
    fecha_generacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_reunion) REFERENCES Reunion(id_reunion) ON DELETE CASCADE
);

-- Tablas relacionadas con Tester

-- Tabla Prueba (catálogo de pruebas)
CREATE TABLE Prueba (
    id_prueba SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    tipo VARCHAR(50) CHECK (tipo IN ('Unitaria', 'Integración', 'Sistema', 'Aceptación', 'Rendimiento', 'Seguridad'))
);

-- Tabla que relaciona Tester con Prueba (muchos a muchos con aptitudes)
CREATE TABLE Tester_Prueba (
    id_tester INTEGER,
    id_prueba INTEGER,
    nivel_aptitud VARCHAR(30) CHECK (nivel_aptitud IN ('Bajo', 'Medio', 'Alto', 'Experto')),
    PRIMARY KEY (id_tester, id_prueba),
    FOREIGN KEY (id_tester) REFERENCES Tester(id_empleado) ON DELETE CASCADE,
    FOREIGN KEY (id_prueba) REFERENCES Prueba(id_prueba) ON DELETE CASCADE
);

-- Tabla PlanPrueba (plan para implementar pruebas)
CREATE TABLE PlanPrueba (
    id_plan SERIAL PRIMARY KEY,
    id_prueba INTEGER NOT NULL,
    id_proyecto INTEGER NOT NULL,
    id_tester INTEGER,
    fecha_inicio DATE,
    fecha_fin DATE,
    estado VARCHAR(30) CHECK (estado IN ('Planificado', 'En progreso', 'Completado', 'Cancelado')),
    resultado TEXT,
    FOREIGN KEY (id_prueba) REFERENCES Prueba(id_prueba) ON DELETE CASCADE,
    FOREIGN KEY (id_proyecto) REFERENCES Proyecto(id_proyecto) ON DELETE CASCADE,
    FOREIGN KEY (id_tester) REFERENCES Tester(id_empleado) ON DELETE SET NULL
);

-- Funciones y Triggers

-- Función para verificar que el rol_general coincida con la especialización
CREATE OR REPLACE FUNCTION verificar_rol()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.rol_general = 'Programador' AND 
       NOT EXISTS (SELECT 1 FROM Programador WHERE id_empleado = NEW.id_empleado) THEN
        RAISE EXCEPTION 'Un empleado con rol Programador debe estar en la tabla Programador';
    ELSIF NEW.rol_general = 'Analista' AND 
          NOT EXISTS (SELECT 1 FROM Analista WHERE id_empleado = NEW.id_empleado) THEN
        RAISE EXCEPTION 'Un empleado con rol Analista debe estar en la tabla Analista';
    ELSIF NEW.rol_general = 'Tester' AND 
          NOT EXISTS (SELECT 1 FROM Tester WHERE id_empleado = NEW.id_empleado) THEN
        RAISE EXCEPTION 'Un empleado con rol Tester debe estar en la tabla Tester';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para verificar el rol después de una actualización
CREATE TRIGGER verificar_rol_empleado
AFTER UPDATE OF rol_general ON Empleado
FOR EACH ROW
EXECUTE FUNCTION verificar_rol();

-- Función para verificar que un director pertenezca al departamento que dirige
CREATE OR REPLACE FUNCTION verificar_director_departamento()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.id_director IS NOT NULL AND 
       NOT EXISTS (SELECT 1 FROM Empleado WHERE id_empleado = NEW.id_director AND id_departamento = NEW.id_departamento) THEN
        UPDATE Empleado SET id_departamento = NEW.id_departamento WHERE id_empleado = NEW.id_director;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para verificar que el director pertenezca al departamento
CREATE TRIGGER verificar_director
AFTER INSERT OR UPDATE ON Departamento
FOR EACH ROW
EXECUTE FUNCTION verificar_director_departamento();

-- Inserción de datos de ejemplo

-- Insertar departamentos
INSERT INTO Departamento (nombre, descripcion) VALUES 
('Desarrollo', 'Departamento de desarrollo de software'),
('Análisis', 'Departamento de análisis de sistemas'),
('QA', 'Departamento de control de calidad');

-- Insertar empleados
INSERT INTO Empleado (nombre_completo, rol_general, email, telefono, fecha_contratacion, id_departamento) VALUES 
('Juan Pérez', 'Director', 'juan.perez@incubadora.com', '555-1234', '2020-01-10', 1),
('María Gómez', 'Programador', 'maria.gomez@incubadora.com', '555-2345', '2020-02-15', 1),
('Carlos Rodríguez', 'Programador', 'carlos.rodriguez@incubadora.com', '555-3456', '2020-03-20', 1),
('Ana Martínez', 'Analista', 'ana.martinez@incubadora.com', '555-4567', '2020-04-25', 2),
('Luis Sánchez', 'Analista', 'luis.sanchez@incubadora.com', '555-5678', '2020-05-30', 2),
('Elena Díaz', 'Tester', 'elena.diaz@incubadora.com', '555-6789', '2020-06-05', 3),
('Pedro Fernández', 'Tester', 'pedro.fernandez@incubadora.com', '555-7890', '2020-07-10', 3),
('Laura Torres', 'Director', 'laura.torres@incubadora.com', '555-8901', '2020-08-15', 2),
('Miguel Ortiz', 'Director', 'miguel.ortiz@incubadora.com', '555-9012', '2020-09-20', 3);

-- Actualizar directores de departamentos
UPDATE Departamento SET id_director = 1 WHERE id_departamento = 1;
UPDATE Departamento SET id_director = 8 WHERE id_departamento = 2;
UPDATE Departamento SET id_director = 9 WHERE id_departamento = 3;

-- Insertar programadores
INSERT INTO Programador (id_empleado, lenguaje_dominante, años_experiencia, nivel) VALUES 
(2, 'Java', 5, 'Senior'),
(3, 'Python', 3, 'Semi-senior');

-- Insertar analistas
INSERT INTO Analista (id_empleado, especialidad, años_experiencia, nivel) VALUES 
(4, 'Sistemas', 7, 'Senior'),
(5, 'Negocios', 4, 'Semi-senior');

-- Insertar testers
INSERT INTO Tester (id_empleado, tipo_testing, certificaciones_qa, años_experiencia) VALUES 
(6, 'Automatizado', TRUE, 4),
(7, 'Manual', FALSE, 2);

-- Insertar proyectos
INSERT INTO Proyecto (nombre, descripcion, fecha_inicio, fecha_fin, estado, presupuesto) VALUES 
('CRM Empresarial', 'Sistema CRM para empresas medianas', '2023-01-01', '2023-12-31', 'En progreso', 50000.00),
('App Móvil Finanzas', 'Aplicación móvil para gestión financiera personal', '2023-03-15', '2023-10-15', 'En progreso', 30000.00),
('Portal Web Educativo', 'Portal web para institución educativa', '2023-05-01', '2024-02-28', 'Planificación', 45000.00);

-- Asignar empleados a proyectos
INSERT INTO Empleado_Proyecto (id_empleado, id_proyecto, fecha_asignacion, rol_en_proyecto) VALUES 
(1, 1, '2023-01-01', 'Coordinador'),
(2, 1, '2023-01-01', 'Desarrollador backend'),
(3, 1, '2023-01-15', 'Desarrollador frontend'),
(4, 1, '2023-01-05', 'Analista de sistemas'),
(6, 1, '2023-02-01', 'Tester principal'),
(2, 2, '2023-03-15', 'Desarrollador móvil'),
(5, 2, '2023-03-20', 'Analista de negocios'),
(7, 2, '2023-04-01', 'Tester de usabilidad'),
(3, 3, '2023-05-01', 'Desarrollador fullstack'),
(4, 3, '2023-05-10', 'Analista de requerimientos'),
(5, 3, '2023-05-15', 'Analista de interfaces'),
(6, 3, '2023-06-01', 'Tester de integración');

-- Insertar herramientas de análisis
INSERT INTO HerramientaAnalisis (nombre, tipo, descripcion) VALUES 
('Jira', 'Gestión', 'Herramienta para gestión de proyectos y seguimiento de problemas'),
('Confluence', 'Documentación', 'Herramienta para documentación y colaboración'),
('Lucidchart', 'Diagramación', 'Herramienta para crear diagramas y mockups'),
('Power BI', 'Análisis de datos', 'Herramienta para análisis y visualización de datos'),
('Figma', 'Diseño', 'Herramienta para diseño de interfaces');

-- Asignar herramientas a analistas
INSERT INTO Analista_Herramienta (id_analista, id_herramienta, nivel_dominio) VALUES 
(4, 1, 'Experto'),
(4, 2, 'Avanzado'),
(4, 3, 'Intermedio'),
(5, 1, 'Avanzado'),
(5, 4, 'Experto'),
(5, 5, 'Avanzado');

-- Insertar certificaciones para analistas
INSERT INTO Certificacion (id_analista, nombre, entidad_emisora, fecha_obtencion, fecha_vencimiento) VALUES 
(4, 'Certified Business Analysis Professional (CBAP)', 'IIBA', '2019-05-15', '2022-05-15'),
(4, 'Professional Scrum Product Owner', 'Scrum.org', '2020-03-20', NULL),
(5, 'Microsoft Certified: Power BI Data Analyst Associate', 'Microsoft', '2021-07-10', '2023-07-10'),
(5, 'Certified ScrumMaster (CSM)', 'Scrum Alliance', '2020-06-05', '2022-06-05');

-- Insertar clientes
INSERT INTO Cliente (tipo, nombre, email, telefono, fecha_registro) VALUES 
('Particular', 'Roberto Vázquez', 'roberto.vazquez@email.com', '555-0123', '2022-10-15'),
('Empresa', 'TechSolutions Inc.', 'contacto@techsolutions.com', '555-1234', '2022-09-01'),
('Empresa', 'Educatech', 'info@educatech.org', '555-2345', '2023-02-10');

-- Insertar particulares
INSERT INTO Particular (id_cliente, apellidos, documento_identidad, fecha_nacimiento) VALUES 
(1, 'Vázquez López', '12345678X', '1980-07-20');

-- Insertar empresas
INSERT INTO Empresa (id_cliente, razon_social, nif, sector, contacto_principal) VALUES 
(2, 'TechSolutions Incorporated S.L.', 'B12345678', 'Tecnología', 'Sofía Álvarez'),
(3, 'Educatech Foundation', 'G87654321', 'Educación', 'Manuel García');

-- Insertar reuniones
INSERT INTO Reunion (fecha_hora, duracion, ubicacion, tema, id_analista, id_cliente, id_proyecto) VALUES 
('2023-02-10 10:00:00', 60, 'Sala de Juntas 1', 'Requerimientos iniciales CRM', 4, 2, 1),
('2023-02-17 15:30:00', 90, 'Virtual - Zoom', 'Revisión de mockups CRM', 4, 2, 1),
('2023-03-20 11:00:00', 60, 'Oficina cliente', 'Presentación de prototipo app móvil', 5, 1, 2),
('2023-05-05 09:30:00', 120, 'Sala de Reuniones 2', 'Análisis de requerimientos portal educativo', 4, 3, 3),
('2023-05-20 14:00:00', 60, 'Virtual - Teams', 'Revisión de arquitectura portal educativo', 5, 3, 3);

-- Insertar reportes de reuniones
INSERT INTO ReportesReunion (id_reunion, tipo_reporte, contenido, fecha_generacion) VALUES 
(1, 'Requisitos', 'Listado de requisitos funcionales y no funcionales para el CRM', '2023-02-10 12:15:00'),
(1, 'Acta', 'Acta de la reunión con los puntos tratados y acuerdos', '2023-02-10 18:30:00'),
(2, 'Feedback', 'Retroalimentación sobre los mockups presentados', '2023-02-17 17:45:00'),
(3, 'Demostración', 'Resultados de la demostración del prototipo', '2023-03-20 13:00:00'),
(4, 'Requisitos', 'Documento de requisitos para el portal educativo', '2023-05-05 12:30:00'),
(5, 'Técnico', 'Documento técnico sobre la arquitectura propuesta', '2023-05-20 16:15:00');

-- Insertar pruebas (catálogo)
INSERT INTO Prueba (nombre, descripcion, tipo) VALUES 
('Prueba de inicio de sesión', 'Verificación del proceso de autenticación', 'Unitaria'),
('Prueba de integración de módulos', 'Comprobación de la correcta integración entre módulos', 'Integración'),
('Prueba de carga', 'Evaluación del rendimiento bajo carga', 'Rendimiento'),
('Prueba de usabilidad', 'Evaluación de la facilidad de uso', 'Aceptación'),
('Prueba de seguridad', 'Verificación de vulnerabilidades', 'Seguridad'),
('Prueba de compatibilidad', 'Evaluación con distintos navegadores/dispositivos', 'Sistema');

-- Asignar pruebas a testers según sus aptitudes
INSERT INTO Tester_Prueba (id_tester, id_prueba, nivel_aptitud) VALUES 
(6, 1, 'Alto'),
(6, 2, 'Alto'),
(6, 3, 'Medio'),
(6, 5, 'Experto'),
(7, 1, 'Medio'),
(7, 4, 'Alto'),
(7, 6, 'Experto');

-- Insertar planes de prueba
INSERT INTO PlanPrueba (id_prueba, id_proyecto, id_tester, fecha_inicio, fecha_fin, estado, resultado) VALUES 
(1, 1, 6, '2023-03-01', '2023-03-05', 'Completado', 'Prueba exitosa con observaciones menores'),
(2, 1, 6, '2023-03-10', '2023-03-15', 'Completado', 'Integración correcta con algunos ajustes pendientes'),
(5, 1, 6, '2023-03-20', '2023-03-25', 'Completado', 'Se detectaron 3 vulnerabilidades de seguridad'),
(4, 2, 7, '2023-05-01', '2023-05-05', 'Completado', 'Buena usabilidad con recomendaciones para mejorar'),
(6, 2, 7, '2023-05-10', '2023-05-15', 'Completado', 'Compatible con la mayoría de dispositivos'),
(1, 3, 6, '2023-07-01', '2023-07-05', 'Planificado', NULL),
(3, 3, 6, '2023-07-10', '2023-07-15', 'Planificado', NULL);

-- Insertar tareas para programadores
INSERT INTO Tarea (descripcion, estado, fecha_asignacion, fecha_vencimiento, id_programador, id_proyecto) VALUES 
('Implementar módulo de autenticación', 'Completada', '2023-01-15', '2023-01-30', 2, 1),
('Desarrollar API REST para clientes', 'Completada', '2023-02-01', '2023-02-20', 2, 1),
('Crear componentes de UI para dashboard', 'En revisión', '2023-02-15', '2023-03-10', 3, 1),
('Integración con sistema de pagos', 'En progreso', '2023-03-01', '2023-03-20', 2, 1),
('Desarrollar interfaz móvil', 'Completada', '2023-03-20', '2023-04-10', 2, 2),
('Implementar sincronización offline', 'En progreso', '2023-04-15', '2023-05-05', 2, 2),
('Crear backend para portal educativo', 'Por hacer', '2023-05-20', '2023-06-15', 3, 3),
('Desarrollar sistema de gestión de cursos', 'Por hacer', '2023-06-01', '2023-06-30', 3, 3);