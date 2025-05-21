import sys
import queue
import threading
import time
import pickle
import struct

import numpy as np

import client_max_rate
import controller_agent

THRESHOLD_FOR_RECENT = 2000

def main():
    controller_address = sys.argv[1]
    controller_port = int(sys.argv[2])
    lb_address = sys.argv[3]
    lb_port = int(sys.argv[4])
    num_request = int(sys.argv[5])

    server_map = controller_agent.listenController(controller_address, controller_port, lambda model_name: 'train' in model_name)
    client_max_rate.run((lb_address, lb_port), server_map, num_request, THRESHOLD_FOR_RECENT)

if __name__ == "__main__":
    main()