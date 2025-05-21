import threading
import time
import statistics
import queue
import pickle
import os
import sys

from common import make_request_sync, make_request_async, prepare_request_binary, AtomicCounter, AtomicCounterRefresh
from model.index import model_map

PENDING_REQUEST_LIMITER = 64

class Request:
    def __init__(self, id, model_name, data_b, start_time):
        self._id = id
        self._model_name = model_name
        self._data_b = data_b
        self._start_time = start_time
        self._end_time = None

    def complete(self, end_time):
        self._end_time = end_time

class RequestGenerator:
    def __init__(self, server_map, num_request):
        self._server_map = server_map
        self._id_generator = 0
        self._model_input = {}

        self._num_request = num_request
        self._total_sent = 0

        self.importDataByte()

    def importDataByte(self):
        # Inference
        request_size = 8
        for key in model_map:
            self._model_input[key] = prepare_request_binary(key, request_size, request_size * 8)
        # Training
        train_model = 'resnet152_train'
        request_size = 96
        self._model_input[train_model] = prepare_request_binary(train_model, request_size, request_size * 8)

    def getRequestID(self):
        self._id_generator += 1
        return self._id_generator

    def getDataBytes(self, model_name):
        try:
            data_b = self._model_input[model_name.split('-')[0]]
            return data_b
        except:
            return None

    def constructRequest(self, model_name, start_time):
        if model_name is None:
            return None
        request = Request(self.getRequestID(), model_name, self.getDataBytes(model_name), start_time)
        self._total_sent += 1
        return request

    def hasNext(self):
        return (self._num_request == 0) or (self._total_sent < self._num_request)

    def next(self):
        raise 'Not Implemented'

class WorkerInfo:
    def __init__(self, id, delay, request_generator: RequestGenerator, sync):
        self._id = id
        self._delay = delay
        self._request_generator = request_generator
        self._sync = sync

pending_counter = AtomicCounter()
total_sent = AtomicCounter()
def increase_pending_counter():
    global pending_counter
    pending_counter.increase()
def decrease_pending_counter():
    global pending_counter
    pending_counter.decrease()
def get_pending_counter():
    global pending_counter
    return pending_counter.get()
def get_total_sent():
    global total_sent
    return total_sent.get()
def increase_total_sent():
    global total_sent
    total_sent.increase()

def thd_loop_func_print_info(qin, threshold_for_recent, stop_flag):
    start_time = int(time.time())
    total_count = 0
    latency_list = []
    time_list = []
    last_time = int(time.time()) + 0.1
    last_count = total_count
    while True:
        try:
            res = qin.get(False)
            latency = (res._end_time - res._start_time) * 1000
            latency_list.append(latency)
            time_list.append(time.time())
            total_count += 1
            decrease_pending_counter()
        except:
            time.sleep(0.001)

        if time.time() - last_time > 1 and len(latency_list) > 0:
            period = time.time() - last_time
            last_time = time.time()
            latency_list = latency_list[-threshold_for_recent:]
            time_list = time_list[-threshold_for_recent:]
            sorted_list = sorted(latency_list)
            len_list = len(sorted_list)

            print ('Real-time throughput', time.time(), last_time, int(time.time()) - start_time, int((total_count - last_count) / period), sep=', ')
            print ('Average Throughput (%d/%d): %d rps' % (len_list, total_count, len_list / (time.time() - time_list[0])))
            print ('Average Latency: %f ms' % (statistics.mean(sorted_list)))
            print ('   99th Latency: %f ms' % (sorted_list[int(0.99 * len_list)]))
            print ('   95th Latency: %f ms' % (sorted_list[int(0.95 * len_list)]))
            print ('   90th Latency: %f ms' % (sorted_list[int(0.90 * len_list)]))
            print ('   75th Latency: %f ms' % (sorted_list[int(0.75 * len_list)]))
            print ('   50th Latency: %f ms' % (sorted_list[int(0.50 * len_list)]))
            print ()
            print ()
            sys.stdout.flush()
            last_count = total_count

        if stop_flag.is_set() and qin.empty() and get_pending_counter() == 0:
            break  # Exit when stop flag is set and no more items in the queue
    print ('Total get:', total_count)
    print ('Print exited')

def thd_loop_func_make_request(end_host, worker_info: WorkerInfo, sending_counter, response_queue):
    time.sleep(worker_info._delay)
    generator = worker_info._request_generator
    make_request = make_request_sync if worker_info._sync else make_request_async
    req_senders = []
    while generator.hasNext():
        if get_pending_counter() > PENDING_REQUEST_LIMITER:
            time.sleep(0.001)
            continue

        request = generator.next()
        if request is None:
            time.sleep(0.001)
            continue
        increase_pending_counter()
        sending_counter.increase()
        increase_total_sent()
        t = make_request(end_host[0], end_host[1], request, response_queue)
        req_senders.append(t)
        
    if len(req_senders) > 0:
        for t in req_senders:
            if t is not None:
                t.join()
    print ('Worker %d finished' % worker_info._id)

def run(end_host, worker_list, threshold_for_recent):
    response_queue = queue.Queue()
    stop_flag = threading.Event()
    t_print = threading.Thread(target=thd_loop_func_print_info, args=(response_queue, threshold_for_recent, stop_flag))
    t_print.start()

    sending_counter = AtomicCounterRefresh('Sending Rate')
    worker_threads = []
    for worker_info in worker_list:
        t_worker = threading.Thread(target=thd_loop_func_make_request, args=(end_host, worker_info, sending_counter, response_queue))
        t_worker.start()
        worker_threads.append(t_worker)
    for t in worker_threads:
        t.join()     
    stop_flag.set()

    t_print.join()
    print ('All threads finished.')
    print ('Total sent:', get_total_sent())
    sys.stdout.flush()
    os._exit(0)