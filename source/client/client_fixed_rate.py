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

class AdaptiveInterval:
    def __init__(self):
        self._target = 1
        self._interval = 1.0
        self._stride = 1e-4
        self._loop = 0

    def reset(self, target):
        self._target = target
        self._interval = 1.0 / target

    def update(self, throughput):
        self._loop += 1
        if self._loop >= 4:
            self._loop = 0
            self._stride /= 2
        if throughput < self._target:
            self._interval -= self._stride
        elif throughput > self._target:
            self._interval += self._stride

    def get(self):
        return self._interval

class RequestGeneratorFixedRate(client_template.RequestGenerator):
    def __init__(self, server_map, throughput_per_gpu, num_request):
        super().__init__(server_map, num_request)
        self._throughput_per_gpu = throughput_per_gpu
        self._adaptive_interval = AdaptiveInterval()
        self._target_queue = queue.Queue()
        self._last_record_time = int(time.time())
        self._sent_reqeust_count = 0
        self._last_request_time = time.time()
        
    def wait(self):
        sleep_time = max(0, self._last_request_time + self._adaptive_interval.get() - time.time())
        time.sleep(sleep_time)
        self._last_request_time = time.time()

    def update_interval(self):
        if time.time() - self._last_record_time > 1.0:
            # print ('Real-time sending rate: %d rps' % self._sent_reqeust_count, self._adaptive_interval._stride)
            self._adaptive_interval.update(self._sent_reqeust_count)
            self._last_record_time = time.time()
            self._sent_reqeust_count = 0
        self._adaptive_interval.reset(len(self._server_map.valid_server_list()) * self._throughput_per_gpu)

    def generate(self):
        model_name = None
        while model_name is None:
            if self._target_queue.empty():
                for server_id in self._server_map.server_list():
                    self._target_queue.put(server_id)
            model_name = self._server_map.get(self._target_queue.get())
        self._sent_reqeust_count += 1
        return super().constructRequest(model_name, time.time())

    def next(self):
        self.wait()
        self.update_interval()
        request = self.generate()
        return request

def run(end_host, server_map: controller_agent.ServerMap, num_request, throughput_per_gpu, threshold_for_recent):
    generator = RequestGeneratorFixedRate(server_map, throughput_per_gpu, num_request)
    worker_info = client_template.WorkerInfo(0, 0, generator, False)
    client_template.run(end_host, [worker_info], threshold_for_recent)