import struct
import threading
import os
import queue
import time
import pickle

from model.index import get_model_module
from py_utils.tcp import TcpClient

class AtomicCounter:
    def __init__(self, name='Default Counter', value = 0):
        self._name = name
        self._value = value
        self._lock = threading.Lock()

    def set(self, value):
        self._lock.acquire()
        self._value = value
        self._lock.release()

    def reset(self):
        self.set(0)

    def increase(self):
        self._lock.acquire()
        self._value += 1
        self._lock.release()

    def decrease(self):
        self._lock.acquire()
        self._value -= 1
        self._lock.release()

    def get(self):
        self._lock.acquire()
        ret = self._value
        self._lock.release()
        return ret

class AtomicCounterRefresh(AtomicCounter):
    def __init__(self, name='Default Counter', value=0):
        super().__init__(name, value)
        self._last_time = time.time()

    def refresh(self):
        if time.time() - self._last_time > 1.0:
            super().reset()
            self._last_time = time.time()

    def increase(self):
        self.refresh()
        super().increase()

    def decrease(self):
        self.refresh()
        super().decrease()

def prepare_request_binary(request_model, request_size, global_batch_size):
    if 'train' in request_model:
        data_ids = [i for i in range(request_size)]
        data_b = pickle.dumps((global_batch_size, data_ids))
    else:
        model_module = get_model_module(request_model)
        images, _ = model_module.import_data(request_size)
        images_b = images.numpy().tobytes()
        data_b = images_b
    return data_b

def import_request_list(filename):
    model_input = {}
    request_list = []
    with open(filename) as f:
        f.readline() # Title
        for line in f.readlines():
            parts = line.split(',')
            request_id = len(request_list)
            request_model = parts[0].strip()
            request_size = int(parts[1].strip())
            request_interval = float(parts[2].strip())
            request_list.append((request_id, request_model, request_size, request_interval))
            if (request_model, request_size) not in model_input:
                model_input[(request_model, request_size)] = prepare_request_binary(request_model, request_size, request_size * 8)
    return request_list, model_input

def make_request_sync(address, port, request, qout: queue.Queue):
    # Connect to service provider
    try:
        lb = TcpClient(address, port)
    except:
        print ('LB failed')
        os._exit(1)
    
    try:
        # send model_name and data
        model_name_b = request._model_name.encode()
        lb.tcpSendWithLength(model_name_b)
        lb.tcpSendWithLength(request._data_b)
        # Get response
        _ = lb.tcpRecvWithLength()
        # Output output
        request.complete(time.time())
        qout.put(request)
    except:
        print ('Request error', request._id)

def make_request_async(address, port, request, qout: queue.Queue):
    t = threading.Thread(target=make_request_sync, args=(address, port, request, qout))
    t.start()
    return t