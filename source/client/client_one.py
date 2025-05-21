import sys
import struct
import queue
import threading
import time
import statistics

import numpy as np
import torch

from source.py_utils.tcp import TcpClient
from common import import_request_list

def main():
    request_list_name = sys.argv[1]
    lb_address = sys.argv[2]
    lb_port = int(sys.argv[3])

    request_list, model_input = import_request_list(request_list_name)
    
    latency_list = []
    for request_id, request_model, request_size, _ in request_list:
        data_b = model_input[(request_model, request_size)]
        
        time_1 = time.time()
        # Connect to LB
        lb = TcpClient(lb_address, lb_port)
        # send model_name and data
        model_name_b = request_model.encode()
        lb.tcpSendWithLength(model_name_b)
        lb.tcpSendWithLength(data_b)
        # Get response
        _ = lb.tcpRecvWithLength()
        time_2 = time.time()

        latency = (time_2 - time_1) * 1000
        # output = torch.from_numpy(np.frombuffer(output_b, dtype=np.float32)).reshape(request_size, -1)
        # print (request_id, request_model, "Latency: %f ms" % latency, output[0].sum().item())
        print (request_id, request_model, "Latency: %f ms" % latency)
        latency_list.append(latency)

    print ('Average Latency: %f (%f)' % (statistics.mean(latency_list), statistics.stdev(latency_list)))

if __name__ == "__main__":
    main()