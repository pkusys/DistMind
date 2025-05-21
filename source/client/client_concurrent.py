import sys
import struct
import queue
import threading
import time
import statistics

import numpy as np
import torch

from common import import_request_list
from model.index import get_model_module
from source.py_utils.tcp import TcpClient

def make_request(lb_address, lb_port, qin: queue.Queue, qout: queue.Queue):
    while not qin.empty():
        request_id, request_model, data_b = qin.get()

        time_1 = time.time()
        # Connect to LB
        lb = TcpClient(lb_address, lb_port)
        # send model_name and data
        model_name_b = request_model.encode()
        lb.tcpSendWithLength(model_name_b)
        lb.tcpSendWithLength(data_b)
        # Get response
        output_b = lb.tcpRecvWithLength()
        time_2 = time.time()

        qout.put((request_id, time_1, time_2, output_b))

def main():
    request_list_name = sys.argv[1]
    lb_address = sys.argv[2]
    lb_port = int(sys.argv[3])
    concurrent = int(sys.argv[4])

    request_list, model_input = import_request_list(request_list_name)
    q_request = queue.Queue()
    q_response = queue.Queue()

    for request_id, request_model, request_size, _ in request_list:
        data_b = model_input[(request_model, request_size)]
        q_request.put((request_id, request_model, data_b))
    
    threads = []
    for _ in range(concurrent):
        t = threading.Thread(target=make_request, args=(lb_address, lb_port, q_request, q_response))
        t.start()
        threads.append(t)

    latency_list = []
    for _ in range(len(request_list)):
        request_id, time_1, time_2, output_b = q_response.get()
        latency = (time_2 - time_1) * 1000
        output = torch.from_numpy(np.frombuffer(output_b, dtype=np.float32)).reshape(request_size, -1)
        print (request_id, "Latency: %f ms" % latency, output[0].sum().item())
        latency_list.append(latency)
    print ('Average Latency: %f (%f)' % (statistics.mean(latency_list), statistics.stdev(latency_list)))

if __name__ == "__main__":
    main()