import socket
from datetime import datetime

def conectar_cliente(servidor_ip, puerto):
    cliente = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        print(f"[Cliente] Intentando conectar a {servidor_ip}:{puerto}")
        cliente.connect((servidor_ip, puerto))
        
        hora_conexion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Cliente] Conectado exitosamente al servidor a las {hora_conexion}")
        print(f"[Cliente] Puedes escribir mensajes (escribe 'salir' para terminar)")
        print("-" * 50)
        
        while True:
            mensaje = input("[Cliente] Escribe tu mensaje: ")
            
            if mensaje.lower() == 'salir':
                print("[Cliente] Cerrando conexión...")
                break
            
            if mensaje.strip() == "":
                print("[Cliente] Mensaje vacío, intenta de nuevo")
                continue
            
            # Agregar timestamp al mensaje
            hora_envio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mensaje_con_timestamp = f"[{hora_envio}] {mensaje}"
            
            try:
                cliente.sendall(mensaje_con_timestamp.encode())
                respuesta = cliente.recv(1024)
                
                print(f"[Cliente] Respuesta del servidor: {respuesta.decode()}")
                print("-" * 30)
                
            except Exception as e:
                print(f"[Cliente] Error al enviar mensaje: {e}")
                break
                
    except ConnectionRefusedError:
        print(f"[Cliente] Error: No se pudo conectar al servidor {servidor_ip}:{puerto}")
        print("[Cliente] Verifica que el servidor esté ejecutándose y el puerto esté abierto")
    except Exception as e:
        print(f"[Cliente] Error de conexión: {e}")
    finally:
        cliente.close()
        print("[Cliente] Conexión cerrada")

if __name__ == "__main__":
    try:
        print("=== CLIENTE DE MENSAJERÍA ===")
        IP_SERVIDOR = input("IP del servidor: ")
        PUERTO = int(input("Puerto del servidor: "))
        
        conectar_cliente(IP_SERVIDOR, PUERTO)
        
    except ValueError:
        print("[Cliente] Error: El puerto debe ser un número")
    except KeyboardInterrupt:
        print("\n[Cliente] Cliente cerrado por el usuario")