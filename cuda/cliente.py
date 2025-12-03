import socket
from datetime import datetime

def obtener_hora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def enviar_a_coordinador(mensaje, host="127.0.0.1", puerto=5000):
    try:
        cliente = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cliente.connect((host, puerto))
        
        print(f"[{obtener_hora()}] Enviando mensaje al coordinador: {mensaje}")
        cliente.sendall(mensaje.encode())

        respuesta = cliente.recv(4096).decode()
        print(f"[{obtener_hora()}] Respuesta del coordinador:\n{respuesta}")
        
        cliente.close()
    except Exception as e:
        print(f"[{obtener_hora()}] Error al conectar con el coordinador: {e}")

if __name__ == "__main__":
    print("Cliente conectado al coordinador. Presiona ENTER sin texto para salir.\n")
    while True:
        mensaje = input("Mensaje a enviar: ")
        if mensaje.strip() == "":
            break
        enviar_a_coordinador(mensaje)

