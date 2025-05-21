import threading
import struct
import time

from py_utils.tcp import TcpClient


class ServerMap:
    def __init__(self):
        self._map = {}
        self._lock = threading.Lock()

    def set(self, server_id, model_name):
        self._lock.acquire()
        self._map[server_id] = model_name
        self._lock.release()

    def get(self, server_id):
        self._lock.acquire()
        ret = self._map[server_id] if server_id in self._map else None
        self._lock.release()
        return ret

    def server_list(self):
        return self._map.keys()

    def valid_server_list(self):
        return [key for key, value in self._map.items() if value is not None]

class ControllerAgent:
    def __init__(self, ctrl_addr, ctrl_port, filter=None, callback=None):
        self._client = TcpClient(ctrl_addr, ctrl_port)
        self._filter = filter
        self._callback = callback
        self._model_list = None
        self._model_distribution = None
        self._server_map = ServerMap()

    def initialize(self):
        self._model_list = []
        self._model_distribution = []

        num_model_b = self._client.tcpRecv(8)
        num_model = struct.unpack('Q', num_model_b)[0]
        for _ in range(num_model):
            model_name = self._client.tcpRecvWithLength().decode()
            model_prob = struct.unpack('d', self._client.tcpRecv(8))[0]
            self._model_list.append(model_name)
            self._model_distribution.append(model_prob)

    def update(self):
        server_id_b = self._client.tcpRecv(8)
        server_ip, server_port = struct.unpack('II', server_id_b)
        server_id = (server_ip << 32) + server_port
        model_name = self._client.tcpRecvWithLength().decode()
        if self._filter is not None:
            model_name = model_name if self._filter(model_name) else None
        self._server_map.set(server_id, model_name)
        self._client.tcpSend(b'abcd')
        if model_name is not None and self._callback is not None:
            self._callback(server_ip, server_port, model_name)

def listenController(strl_addr, ctrl_port, filter=None, callback=None):
    controller = ControllerAgent(strl_addr, ctrl_port, filter, callback)
    controller.initialize()

    def loop_update_server_map():
        while True:
            controller.update()
    t_server_map = threading.Thread(target=loop_update_server_map)
    t_server_map.start()
    time.sleep(1)
    return controller._server_map