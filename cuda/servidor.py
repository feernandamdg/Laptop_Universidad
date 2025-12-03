import socket
import threading
from datetime import datetime

def manejar_cliente(conexion, direccion):
    print(f"[Servidor] Conexión desde {direccion}")
    while True:
        try:
            datos = conexion.recv(1024)
            if not datos:
                break
            mensaje = datos.decode()
            hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Servidor {direccion}] {hora_actual} - Recibido: {mensaje}")
            respuesta = f"[{hora_actual}] Respuesta del servidor {direccion}: {mensaje}"
            conexion.sendall(respuesta.encode())
        except:
            break
    conexion.close()

def iniciar_servidor(puerto):
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.bind(('0.0.0.0', puerto))
    servidor.listen()
    print(f"[Servidor] Escuchando en puerto {puerto}")
    while True:
        conn, addr = servidor.accept()
        hilo = threading.Thread(target=manejar_cliente, args=(conn, addr))
        hilo.start()

if __name__ == "__main__":
    PUERTO = int(input("Puerto del servidor: "))  # Por ejemplo, 6000 o 6001
    iniciar_servidor(PUERTO)
