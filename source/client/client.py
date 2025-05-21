import sys
import struct
import queue
import threading
import time
import statistics

import numpy as np
import torch

from model.index import get_model_module
from source.py_utils.tcp import TcpClient
from common import import_request_list

def make_request(request_id, request_model, request_size, data_b, address, port, start_time, qout: queue.Queue):
    # Connect to LB
    lb = TcpClient(address, port)
    
    # send model_name and data
    model_name_b = request_model.encode()
    lb.tcpSendWithLength(model_name_b)
    lb.tcpSendWithLength(data_b)

    # Get response
    output_b = lb.tcpRecvWithLength()
    output = torch.from_numpy(np.frombuffer(output_b, dtype=np.float32)).reshape(request_size, -1)
    # print (request_id, request_model, (time.time() - start_time) * 1000, output[0].sum().item())
    if request_id % 32 == 0:
        print (request_id, request_model, (time.time() - start_time) * 1000)

    # Output output
    qout.put((request_id, request_model, output, start_time, time.time()))

def print_info(qin, n_req):
    latency_list = []
    for _ in range(n_req):
        request_id, request_model, output, start_time, end_time = qin.get()
        latency = (end_time - start_time) * 1000
        latency_list.append(latency)
        if len(latency_list) % 1000 == 0:
            print ('Average Latency (%d): %f' % (len(latency_list), statistics.mean(latency_list)))
    print ('Average Latency: %f (%f)' % (statistics.mean(latency_list), statistics.stdev(latency_list)))

def main():
    request_list_name = sys.argv[1]
    lb_address = sys.argv[2]
    lb_port = int(sys.argv[3])

    request_list, model_input = import_request_list(request_list_name)

    response_queue = queue.Queue()
    t_info = threading.Thread(target=print_info, args=(response_queue, len(request_list)))
    t_info.start()

    last_time = 0.0
    for request_id, request_model, request_size, interval in request_list:
        data_b = model_input[(request_model, request_size)]
        if time.time() < last_time + interval:
            time.sleep(max(0, last_time + interval - time.time()))
        t_req = threading.Thread(target=make_request, args=(request_id, request_model, request_size, data_b, lb_address, lb_port, time.time(), response_queue))
        t_req.start()
        last_time = time.time()

    # latency_list = []
    # for _ in range(len(request_list)):
    #     request_id, request_model, output, start_time, end_time = response_queue.get()
    #     latency = (end_time - start_time) * 1000
    #     latency_list.append(latency)
    #     if len(latency_list) % 1000 == 0:
    #         print ('Average Latency (%d): %f' % (len(latency_list), statistics.mean(latency_list)))
    
    # print ('Average Latency: %f (%f)' % (statistics.mean(latency_list), statistics.stdev(latency_list)))
    t_info.join()

if __name__ == "__main__":
    main()