import socket
import struct

class TcpAgent:
    def __init__(self, conn_fd):
        self._conn_fd = conn_fd

    def __del__(self):
        self._conn_fd.close()

    def tcpSend(self, msg_b):
        self._conn_fd.sendall(msg_b)

    def tcpSendWithLength(self, msg_b):
        # print ('tcpSendWithLength', 1, len(msg_b))
        msg_len = len(msg_b)
        msg_len_b = struct.pack('Q', msg_len)
        # print ('tcpSendWithLength', 2, len(msg_len_b))
        self.tcpSend(msg_len_b)
        # print ('tcpSendWithLength', 3)
        self.tcpSend(msg_b)
        # print ('tcpSendWithLength', 4)

    def tcpRecv(self, msg_length):
        return self._conn_fd.recv(msg_length, socket.MSG_WAITALL)

    def tcpRecvWithLength(self):
        try:
            msg_len_b = self.tcpRecv(8)
            msg_len = struct.unpack('Q', msg_len_b)[0]
            msg_b = self.tcpRecv(msg_len)
        except Exception as e:
            print('tcpRecvWithLength error', e)
            msg_b = None
        return msg_b

class TcpClient(TcpAgent):
    def __init__(self, address, port):
        super().__init__(None)
        self._conn_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn_fd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn_fd.connect((address, port))

class TcpServer:
    def __init__(self, address, port):
        self._address = address
        self._port = port
        self._server_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_fd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._server_fd.bind((self._address, self._port))
        self._server_fd.listen(100)

    def __del__(self):
        self._server_fd.close()

    def tcpAccept(self):
        conn, _ = self._server_fd.accept()
        return TcpAgent(conn)