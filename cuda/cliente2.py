import socket 
# configuración del cliente 
HOST = '127.0.0.1'  # localhost 
PORT = 6666        # puerto del servidor 
 
# creación del socket 
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
    s.connect((HOST, PORT)) 
    s.sendall(b'Hola, servidor!') 
    data = s.recv(1024) 

print('Recibido:', repr(data)) 