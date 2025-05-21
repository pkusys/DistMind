import sys
import json
import time
import struct
import threading
import queue
import argparse
import numpy as np

from tcp import TcpServer, TcpAgent

SIGNAL_CACHE_IN = 0
SIGNAL_CACHE_OUT = 1
SIGNAL_CACHE_REPLY = 2

KVSTORAGE_OP_READ = 0
KVSTORAGE_OP_WRITE = 1
KVSTORAGE_OP_ACK = 2
MEMORY_MANAGER_AMPLIFIER = 1.2

class DivisionForTrainAndInference:
    def __init__(self, gpu_list):
        self.queue_for_train = queue.Queue()
        self.queue_for_inference = queue.Queue()
        self.set_for_train = set()

        for gpu in sorted(gpu_list):
            self.queue_for_inference.put(gpu)

    def num_train(self):
        return len(self.set_for_train)

    def increase_train(self):
        gpu = self.queue_for_inference.get()
        self.set_for_train.add(gpu)
        self.queue_for_train.put(gpu)
        return gpu

    def decrease_train(self):
        gpu = self.queue_for_train.get()
        self.set_for_train.remove(gpu)
        self.queue_for_inference.put(gpu)
        return gpu

    def is_train(self, gpu):
        return (gpu in self.set_for_train)


def import_model_list(filename):
    model_list = []
    with open(filename) as f:
        _ = f.readline()
        for line in f.readlines():
            parts = line.split(',')
            model_name = parts[0].strip()
            if 'train' not in model_name:
                model_list.append(model_name)
    return model_list

def generate_zipf_distribution(n, s):
    weights = np.power(1.0 / np.arange(1, n + 1), s)
    distribution = weights / np.sum(weights)
    return list(distribution)

def import_server_list(filename):
    server_map = {}
    with open(filename) as f:
        for line in f.readlines():
            parts = line.split(':')
            ip = parts[0].strip()
            port = int(parts[1].strip())
            server_map['%s:%d' % (ip, port)] = None
    return server_map

def broadcast(agent, server_id, model_name):
    if agent is None:
        return
    if model_name is None:
        return
    agent.tcpSendWithLength(server_id.encode())
    agent.tcpSendWithLength(model_name.encode())
    _ = agent.tcpRecv(4)

def init_broadcast(agent: TcpAgent, server_map):
    for server_id, model_name in server_map.items():
        broadcast(agent, server_id, model_name)

def thd_loop_func_accept_subscriber(addr, port, subscribe_list, subscribe_lock, server_map):
    server = TcpServer(addr, port)
    print ('Accepting subscriber at %s:%d' % (addr, port))
    while True:
        agent = server.tcpAccept()
        subscribe_lock.acquire()
        subscribe_list.append(agent)
        init_broadcast(agent, server_map)
        subscribe_lock.release()

        print ('Accepting subscriber')

def generate_server_queue(server_map):
    tmp = {}
    for server_id in server_map:
        server_ip = server_id.split(':')[0]
        tmp[server_ip] = []
    for server_id in server_map:
        server_ip = server_id.split(':')[0]
        tmp[server_ip].append(server_id)

    ret = queue.Queue()
    flag = True
    while flag:
        flag = False
        for server_ip in tmp:
            if tmp[server_ip] is not None:
                flag = True
                ret.put(tmp[server_ip][-1])
                del tmp[server_ip][-1]
                if len(tmp[server_ip]) == 0:
                    tmp[server_ip] = None
    return ret

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help='Configuration file')
    argparser.add_argument('--use_train', type=bool, default=False, help='Use training or inference')
    argparser.add_argument('--zipf_s', type=float, default=0.9, help='Zipf distribution parameter')
    argparser.add_argument('--rescheduling_period', type=float,default=1.0, help='Rescheduling period')
    args = argparser.parse_args()

    # Import configuration
    configuration_file = args.config
    configuration = json.load(open(configuration_file))
    print ('Import configuration')
    # print (configuration)

    use_train = args.use_train
    print ('Use train', use_train)

    # Import model list
    model_list = import_model_list(configuration['model_list_filename'])
    model_distribution = generate_zipf_distribution(len(model_list), args.zipf_s)
    print ('Import model list')
    # print (model_distribution)

    # Read the server list from files
    server_map = import_server_list(configuration['server_list_file'])
    print ('Import server list')
    # print (server_map)

    # Accept the client
    subscribe_list = []
    subscribe_lock = threading.Lock()
    thd_loop_accept_client = threading.Thread(target=thd_loop_func_accept_subscriber, args=(
        configuration['addr_for_subscriber'],
        configuration['port_for_subscriber'],
        subscribe_list,
        subscribe_lock,
        server_map
    ))
    thd_loop_accept_client.start()
    time.sleep(1)
    print ()
    
    # Initialize models
    for server_id in server_map:
        server_map[server_id] = model_list[0]
        for index, agent in enumerate(subscribe_list):
            try:
                broadcast(agent, server_id, model_list[0])
            except:
                print ("Lost client is handled")
                subscribe_list[index] = None

    # Change models periodically
    print ('Start', file=sys.stderr)
    inference_workload = configuration['inference_workload_s']
    server_queue = generate_server_queue(server_map)
    division = DivisionForTrainAndInference([server_id for server_id in server_map])
    while True:
        start_time = time.time()
        workload_tag = int(time.time()) % len(inference_workload)
        expected_train_workload = len(server_map) - inference_workload[workload_tag]

        if use_train:
            if division.num_train() < expected_train_workload:
                server_id = division.increase_train()
                model_name = 'resnet152_train'
                rescheduling_period = 0.0001
            elif division.num_train() > expected_train_workload:
                server_id = division.decrease_train()
                model_name = np.random.choice(model_list, p=model_distribution)
                rescheduling_period = 0.0001
            elif server_queue.empty():
                time.sleep(0.001)
                continue
            else:
                server_id = server_queue.get()
                server_queue.put(server_id)
                if division.is_train(server_id):
                    time.sleep(0.001)
                    continue
                model_name = np.random.choice(model_list, p=model_distribution)
                rescheduling_period = args.rescheduling_period
        else:
            if server_queue.empty():
                time.sleep(0.001)
                continue
            else:
                server_id = server_queue.get()
                server_queue.put(server_id)

            model_name = np.random.choice(model_list, p=model_distribution)
            rescheduling_period = args.rescheduling_period

        print ('Update', server_id, model_name)
        sys.stdout.flush()
        # Update model and broadcast
        subscribe_lock.acquire()
        server_map[server_id] = model_name
        for index, agent in enumerate(reversed(subscribe_list)):
            try:
                broadcast(agent, server_id, model_name)
            except:
                print ("Lost client is handled")
                subscribe_list[index] = None
        subscribe_lock.release()

        # Wait for the next period
        elapsed_time = time.time() - start_time
        sleep_time = max(0, rescheduling_period - elapsed_time)
        time.sleep(sleep_time)

        # for server_id in server_map:
        #     print (server_id, server_map[server_id])
        # print ()

if __name__ == "__main__":
    main()