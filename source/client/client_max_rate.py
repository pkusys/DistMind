import sys
import struct
import queue
import threading
import time
import statistics
import pickle
import os

import numpy as np

from model.index import model_map
from py_utils.tcp import TcpClient
import common
import client_template
import controller_agent
from client_template import Request

class RequestGeneratorMaxRate(client_template.RequestGenerator):
    def __init__(self, server_map, server_id, num_request):
        super().__init__(server_map, num_request)
        self._server_id = server_id
        
    def next(self):
        model_name = self._server_map.get(self._server_id)
        return super().constructRequest(model_name, time.time())

def run(end_host, server_map: controller_agent.ServerMap, num_request, threshold_for_recent):
    worker_list = []
    for index, server_id in enumerate(server_map.server_list()):
        generator = RequestGeneratorMaxRate(server_map, server_id, num_request)
        worker_info = client_template.WorkerInfo(server_id, index + 1, generator, True)
        worker_list.append(worker_info)
    client_template.run(end_host, worker_list, threshold_for_recent)