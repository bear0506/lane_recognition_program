# 서버
import socket, time

host = 'localhost' 
port = 3333 

# 서버소켓 오픈
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host, port))

# 클라이언트 접속 준비 완료
server_socket.listen()

print('echo server start')

#  클라이언트 접속 기다리며 대기 
client_soc, addr = server_socket.accept()

print('connected client addr:', addr)

# 클라이언트가 보낸 패킷 계속 받아 에코메세지 돌려줌
while True:
    data = client_soc.recv(100)
    msg = data.decode() 
    print('recv msg:', msg)
    client_soc.sendall(msg.encode(encoding='utf-8')) 
    if msg == '/end':
        break

time.sleep(5)
print('서버 종료')
server_socket.close() # 사용했던 서버 소켓을 닫아줌