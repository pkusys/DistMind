""" GPU server agent manage multiple GPUs
maintian connection to LB, accept tasks from LB
mange models 
launch MPSClient process 
dispatch task to MPS Clients
"""

import nvgpu
import source.py_utils.tcp as tcp
import socket
import struct
import source.mps.gpu_worker as worker
import torch.multiprocessing as mp
import os
import torch
from model.index import get_model_module
import argparse
import threading
import logging
import time

log_dir = None
training_log_dir = None

def get_gpus():
    """ get GPU ids
    """
    ids = []
    for item in nvgpu.gpu_info():
        ids.append(item['uuid'])
    return ids


def _handleResponseProc(response_q, tcp2LB: tcp.TcpAgent, respLock: mp.Lock, exit_signal):
    logging.debug('started _handleResponseProf')
    while exit_signal.value < 1:
        if not response_q.empty():
            ids, outputs = response_q.get()
            with respLock:
                tcp2LB.tcpSendWithLength(ids[0].encode())  # req_id
                tcp2LB.tcpSendWithLength(ids[1].encode())  # cache_id
                tcp2LB.tcpSendWithLength(outputs)  # actual results

            logging.debug(
                'get a response from %s for request id %s', ids[1], ids[0])

        time.sleep(0.002)
    logging.debug('exiting _handleResponseProc')


def _launchMPSClient(gpu_id, model_name, model_size, tcp2LB, respLock):
    """"""
    env = os.environ.copy()
    task_q = mp.Queue()
    res_q = mp.Queue()
    # is_inf = True if not 'train' in model_name else False
    # one_step_training = None if is_inf else training_fn
    proc = mp.Process(target=worker.workerProc,
                      kwargs={'gpu_id': gpu_id,
                              'model_name': model_name,
                              'model_size': model_size,
                              'is_inf': True,
                              'task_q': task_q,
                              'res_q': res_q,
                              'env': env,
                              'log_dir': log_dir,
                              'one_step_train_fn': None})
    proc.start()

    exit_s = mp.Value('i', 0)
    res_proc = mp.Process(target=_handleResponseProc, args=(
        res_q, tcp2LB, respLock, exit_s))
    res_proc.start()
    return task_q, proc, res_proc, exit_s

def _launchDistTrainingDaemon(ctl_val):
    """ launch a fixed resnet152 training
    init a control mp.Value('i', 0) -> control whether to training or stop
    launch a special process, worker.trainingProc

    """
    gpu_ids = get_gpus()
    env = os.environ.copy()
    world_size = len(gpu_ids)
    for i in range(world_size):
        gpu_id = gpu_ids[i]
        proc = mp.Process(target=worker.trainingProc, args=(gpu_id, env, ctl_val, i, world_size, training_log_dir))
        proc.start()

        logging.debug("launched dist training on gpu %s", gpu_id)

    return None

class WorkerAgent:
    def __init__(self, model_size_file, lb_ip, lb_port):
        """"""
        
        self.task_qs = {}
        self.res_proc_signal = {}
        self.procs = {}
        self.tcp_cli = self._tcpToLB(lb_ip, lb_port)
        self.respLock = mp.Lock()
        self.model_sizes = self._load_model_sizes(model_size_file)
        self.training_proc = {}

    def _load_model_sizes(self, filename):
        ret = {}
        with open(filename, 'r') as ifile:
            for line in ifile:
                name, size = line.strip('\n').split(',')
                ret[name] = int(size)
        return ret

    def run(self):
        """"""
        logging.info('initialized')
        dist_training_init = False
        dist_training_ctl = mp.Value('i', 0)
        while True:
            t = self.tcp_cli.tcpRecvWithLength()
            if t == b'evict':
                cache_id = self.tcp_cli.tcpRecvWithLength().decode()
                self._evictMPSClient(cache_id)
                logging.debug('receive instruction to evit %s', cache_id)
            elif t == b"create":
                gpu_id = self.tcp_cli.tcpRecvWithLength().decode()
                model_name = self.tcp_cli.tcpRecvWithLength().decode()
                cache_id = "{}$${}".format(gpu_id, model_name)
                if cache_id in self.task_qs:
                    logging.debug("cache %s existed", cache_id)
                else:
                    task_q, proc, res_proc, exit_s = _launchMPSClient(gpu_id, model_name, self.model_sizes[model_name],
                                                                    self.tcp_cli, self.respLock)
                    self.task_qs[cache_id] = task_q
                    self.res_proc_signal[cache_id] = exit_s
                    self.procs[cache_id] = [proc, res_proc]
                    logging.debug(
                        'received for creating mps client %s, %s', gpu_id, model_name)
            elif t == b'inf':
                req_id = self.tcp_cli.tcpRecvWithLength().decode()
                cache_id = self.tcp_cli.tcpRecvWithLength().decode()
                data = self.tcp_cli.tcpRecvWithLength()
                # dispatch task; assume the MPS client already there
                if cache_id not in self.task_qs:
                    task_q, proc, res_proc, exit_s = _launchMPSClient(gpu_id, model_name, self.model_sizes[model_name],
                                                                    self.tcp_cli, self.respLock)
                    self.task_qs[cache_id] = task_q
                    self.res_proc_signal[cache_id] = exit_s
                    self.procs[cache_id] = [proc, res_proc]
                    logging.debug("temporal launch cache %s", cache_id)

                self.task_qs[cache_id].put([(req_id, cache_id), data])
                logging.debug("received inf request of %s", cache_id)
            elif t == b'create_train_proc':
                """ start a training proc record to 
                self.training_proc [gpu-id, status-val]
                """
                logging.debug('received signal for creating distributed training proc')
                # gpu_id = self.tcp_cli.tcpRecvWithLength().decode() # fake receive
                if not dist_training_init:
                    _launchDistTrainingDaemon(dist_training_ctl)
                    dist_training_init = True

            elif t == b'start_training':
                """ get a gpu_id, and self.training_proc[gpu-id].val = 1
                """
                dist_training_ctl.value = 1
                logging.debug("start training")
            elif t == b'stop_training':
                """ """
                # gpu_id = self.tcp_cli.tcpRecvWithLength().decode()
                # self.training_proc[gpu_id].value = 0
                dist_training_ctl.value = 0
                logging.debug('stop training')
            else:
                print('unexpected task')

    def _tcpToLB(self, ip, port):
        """"""
        cli = tcp.TcpClient(ip, port)
        gpu_ids = get_gpus()
        # cli.tcpSendWithLength(b"gpu_ids")
        gpus = ";".join(gpu_ids)
        cli.tcpSendWithLength(gpus.encode())
        logging.info('sent gpu ids %s', gpus)
        return cli

    def _evictMPSClient(self, cache_id):
        """"""
        self.task_qs[cache_id].put("exit")
        self.procs[cache_id][0].join()
        logging.debug('worker process closed')
        self.res_proc_signal[cache_id].value = 1
        logging.debug('changed res proc signal to %s',
                      self.res_proc_signal[cache_id].value)
        self.procs[cache_id][1].join()
        del self.procs[cache_id]
        del self.task_qs[cache_id]
        del self.res_proc_signal[cache_id]

        # logging.debug(self.res_proc_signal.keys())


def main():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb-ip", required=True)
    parser.add_argument("--lb-port", required=True, type=int)
    parser.add_argument('--size-list', required=True)
    parser.add_argument('--gpu-num', required=True, type=int)
    parser.add_argument('--ctrl-ip', required=True)
    parser.add_argument('--ctrl-port', required=True, type=int)
    parser.add_argument('--local-ip', required=True)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--training-log-dir', default=None)
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    global log_dir
    global training_log_dir
    log_dir = args.log_dir
    training_log_dir = args.training_log_dir
    if args.debug:
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO)
    agent = WorkerAgent(args.size_list, args.lb_ip, args.lb_port)

    # Register to the controller
    ctrl_address = args.ctrl_ip
    ctrl_port = args.ctrl_port
    address_for_client = args.local_ip
    for gpu_index in range(args.gpu_num):    
        ctrl_client = tcp.TcpClient(ctrl_address, ctrl_port)
        server_id_b = socket.inet_aton(address_for_client) + struct.pack('i', gpu_index + 10000)
        ctrl_client.tcpSend(server_id_b)
        del ctrl_client
        print ('Register to the controller', socket.inet_aton(address_for_client), gpu_index + 10000)
        time.sleep(1)
    
    try:
        agent.run()
    except e:
        print('error', e)
        logging.debug('error', e)


if __name__ == "__main__":
    main()
