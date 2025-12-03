import socket
import threading
from datetime import datetime

def manejar_cliente(conexion, direccion):
    print(f"[Servidor] Nueva conexión desde {direccion}")
    print(f"[Servidor] Conexión establecida a las {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        try:
            datos = conexion.recv(1024)
            if not datos:
                print(f"[Servidor] Cliente {direccion} se desconectó")
                break
            
            mensaje = datos.decode()
            hora_recepcion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[Servidor] [{hora_recepcion}] Mensaje de {direccion}: {mensaje}")
            
            # Preparar respuesta con timestamp
            respuesta = f"[Servidor - {hora_recepcion}] Mensaje recibido: '{mensaje}'"
            conexion.sendall(respuesta.encode())
            
        except Exception as e:
            print(f"[Servidor] Error con cliente {direccion}: {e}")
            break
    
    conexion.close()
    print(f"[Servidor] Conexión con {direccion} cerrada")

def iniciar_servidor(puerto):
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Bind a todas las interfaces (0.0.0.0) para permitir conexiones externas
        servidor.bind(('0.0.0.0', puerto))
        servidor.listen(5)
        
        print(f"[Servidor] Iniciado en puerto {puerto}")
        print(f"[Servidor] Esperando conexiones... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Servidor] Escuchando en todas las interfaces (0.0.0.0:{puerto})")
        
        while True:
            conn, addr = servidor.accept()
            hilo = threading.Thread(target=manejar_cliente, args=(conn, addr))
            hilo.daemon = True
            hilo.start()
            
    except Exception as e:
        print(f"[Servidor] Error al iniciar servidor: {e}")
    finally:
        servidor.close()

if __name__ == "__main__":
    try:
        PUERTO = int(input("Puerto del servidor (ej: 6000): "))
        iniciar_servidor(PUERTO)
    except KeyboardInterrupt:
        print("\n[Servidor] Servidor detenido por el usuario")
    except ValueError:
        print("[Servidor] Error: Puerto debe ser un número")