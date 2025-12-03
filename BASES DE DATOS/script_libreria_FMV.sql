DROP TABLE IF EXISTS usuarios; 
DROP TABLE IF EXISTS roles;
DROP TABLE IF EXISTS libros;
DROP TABLE IF EXISTS autores; 
DROP TABLE IF EXISTS libroAutor;
DROP TABLE IF EXISTS prestamos;
DROP TABLE IF EXISTS devoluciones;
DROP TABLE IF EXISTS multas;
-- --------------------------Creación de tabla Roles (PRIMERO) -----------------
CREATE TABLE roles(
	id_rol SERIAL PRIMARY KEY,
	nombre_rol VARCHAR(50) UNIQUE NOT NULL,
	--Restricciones check
	CONSTRAINT roles_nombre_rol_check CHECK (nombre_rol IN ('administrador', 'bibliotecario', 'lector'))
);

-- ----------Creación de la tabla Usuarios (DESPUÉS de roles)
CREATE TABLE usuarios(
    id_usuario SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    primer_apellido VARCHAR(50) NOT NULL,
    segundo_apellido VARCHAR(50),
    email VARCHAR(100) NOT NULL UNIQUE,
    telefono VARCHAR(20),
    fecha_nacimiento DATE,
    fecha_registro DATE DEFAULT CURRENT_DATE,
    contrasena VARCHAR(50) NOT NULL,
    id_rol INTEGER NOT NULL,
    -- Restricciones clave foránea que referencía a la tabla roles
    CONSTRAINT fk_rol FOREIGN KEY (id_rol) REFERENCES roles(id_rol) ON DELETE CASCADE
);

-- ------------------- Creacion de tabla Autores ---------
CREATE TABLE autores(
	id_autor SERIAL PRIMARY KEY,
	nombre_autor VARCHAR(50) NOT NULL,
	apellido_paterno VARCHAR(50),
	apellido_materno VARCHAR(50)
);

-- ----Creación de tabla libros -------
CREATE TABLE libros(
	id_libro SERIAL PRIMARY KEY,
	ISBN VARCHAR(20) UNIQUE,
	titulo VARCHAR(255) NOT NULL,
	anio_publicacion INTEGER,
	editorial VARCHAR(100),
	edicion INTEGER,
	cantidad_total INTEGER NOT NULL,
	cantidad_disponible INTEGER NOT NULL,
	clasificacion INTEGER NOT NULL,
	-- restricciones CHECK
	CONSTRAINT libros_cantidad_total_check CHECK (cantidad_total >= 0),
	CONSTRAINT libros_check CHECK (cantidad_disponible >= 0 AND cantidad_disponible <= cantidad_total),
	CONSTRAINT libros_clasificacion_check CHECK (clasificacion >= 0 AND clasificacion <= 5)
);

-- ------------------ Creación de tabla LibroAutor --------
CREATE TABLE libroAutor(
	id_libro INTEGER NOT NULL,
	id_autor INTEGER NOT NULL,
	-- Clave primaria compuesta
    PRIMARY KEY (id_libro, id_autor),
	-- restriccion de llaves foraneas que referencian a autores y libros
	CONSTRAINT libroautor_id_autor_fkey FOREIGN KEY (id_autor) REFERENCES autores(id_autor) ON DELETE CASCADE,
	CONSTRAINT libroautor_id_libro_fkey FOREIGN KEY (id_libro) REFERENCES libros(id_libro) ON DELETE CASCADE
);

-- ------------- Creación de tabla Prestamos ---------
CREATE TABLE prestamos(
	id_prestamo SERIAL PRIMARY KEY,
	id_usuario INTEGER,
	id_libro INTEGER, 
	fecha_prestamo DATE NOT NULL,
	fecha_devolucion_max DATE,
	estado_prestamo VARCHAR(20),
	--Restricciones check
	CONSTRAINT prestamos_estado_prestamo_check CHECK (estado_prestamo IN ('pendiente', 'devuelto', 'con retraso')),
	--Restricciones llaves foraneas
	CONSTRAINT prestamos_id_libro_fkey FOREIGN KEY (id_libro) REFERENCES libros(id_libro) ON DELETE CASCADE,
	CONSTRAINT prestamos_id_usuario_fkey FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE CASCADE
);

-- ------------ Creacion de tabla devoluciones ------
CREATE TABLE devoluciones(
	id_devolucion SERIAL PRIMARY KEY,
	--clave foranea que referencia prestamos
	id_prestamo INTEGER NOT NULL, 
	fecha_devolucion DATE NOT NULL,
	-- restricciones llave foranea 
	CONSTRAINT devoluciones_id_prestamo_fkey FOREIGN KEY (id_prestamo) REFERENCES prestamos(id_prestamo) ON DELETE CASCADE
);

-- ---------Creacion de tabla Multas ---------
CREATE TABLE multas(
    id_multa SERIAL PRIMARY KEY,
    id_prestamo INTEGER NOT NULL,
    monto NUMERIC(10,2), -- se actualiza en función del retraso de la devolución
    pagado BOOLEAN DEFAULT FALSE,
    fecha_multa DATE DEFAULT CURRENT_DATE,
    -- Restricción de llave foránea
    CONSTRAINT multas_id_prestamo_fkey FOREIGN KEY(id_prestamo) REFERENCES prestamos(id_prestamo) ON DELETE CASCADE
    -- Nota: Ya no necesitas CONSTRAINT multas_pkey porque SERIAL PRIMARY KEY ya lo hace
);
























