"""
"""
import argparse
import logging
import queue
import random
import sys
import threading as thd
import time
from typing import Dict
from collections import defaultdict

import numpy as np

import source.mps.lru as lru
import source.py_utils.tcp as tcp
from source.controller import controller_agent


class Scheduler:

    def __init__(self, loads, loads_mtx, localLRU, localLRU_mtx, cache_loc, cache_loc_mtx, server_map):
        self.loads = loads
        self.loads_mtx = loads_mtx
        self.localLRU = localLRU
        self.localLRU_mtx = localLRU_mtx
        self.cache_loc = cache_loc # {"model-name": {"gpu-id": qlen, "gpu-id-2": qlen, ...}}
        self.cache_loc_mtx = cache_loc_mtx
        self.server_map = server_map

        self.reqNum = 0
        self.cacheHit = 0
    
    def getInfGPUs(self):
        avaiable_GPUs = self.server_map.valid_server_list()
        GPU_ids = list(self.loads.keys())
        # print('GPU_ids', GPU_ids)
        avaiable_GPUs = list([GPU_ids[i] for i in avaiable_GPUs])
        return avaiable_GPUs
    
    def getCachedGPU(self, model_name):
        """ 
        get the GPU for a cached model with shortest wait queue
        """
        avaiable_GPUs = self.getInfGPUs()

        if model_name in self.cache_loc:
            minQueueGPU = None
            minQueueLen = None
            for gid in self.cache_loc[model_name]:
                qLen = self.cache_loc[model_name][gid]
                assert qLen >= 0
                if minQueueLen is None or qLen < minQueueLen:
                    if gid in avaiable_GPUs:
                        minQueueLen = qLen
                        minQueueGPU = gid
            return (minQueueGPU, minQueueLen)
        else:
            return None

    def updateCache(self, evict, launch):
        if evict:
            gpu_id = evict[0]
            model_name = evict[1]
            del self.cache_loc[model_name][gpu_id]
        if launch:
            model_name = launch[1]
            gpu_id = launch[0]
            if model_name not in self.cache_loc:
                self.cache_loc[model_name] = defaultdict(int)

            self.cache_loc[model_name][gpu_id] += 1

    def getIdleGPUs(self):
        idle = []
        avaiable_GPUs = self.getInfGPUs()
        for gpu in self.loads.keys():
            if self.loads[gpu] < 1 and (gpu in avaiable_GPUs):
                idle.append(gpu)
        return idle

    def _tryPlan(self, model_name):
        """ 
        1. get cache-id with queue-len (cache-id == (gpu-id, model_name)) and idle GPUs
        2. get cache-id with min queue length
        3. if min-queue < 3600 / 100 then put the task to that queue for waiting
           else if idle GPUs not empty then launch one, # can safely evict, because no task not finished
           else wait for next round check 
        """

        with self.cache_loc_mtx and self.loads_mtx and self.localLRU_mtx:
            cachedGPU = self.getCachedGPU(model_name)
            logging.debug("cached GPU %s", cachedGPU)
            # step 1: if there is cache hit with cache queue len < 3600 / 100
            if cachedGPU is not None and cachedGPU[1] is not None:
                # cache hit
                # move cache to recent
                gid, qLen = cachedGPU
                if qLen < 12:
                    lcache = self.localLRU[gid]
                    _ = lcache.get(model_name)
                    # increase cache q job
                    self.cache_loc[model_name][gid] += 1

                    self.cacheHit += 1
                    logging.debug("hit cache, but need to wait for %s jobs", qLen)
                    return gid, [], []

            # step 2: if there is idle GPU
            # comments: no possible to have idle GPU has cache here,
            #           otherwise function returned in previous step, because qLen == 0
            idleGPUs = self.getIdleGPUs()
            if len(idleGPUs) > 0:
                selectedGPU = random.choice(idleGPUs)
                launch = [selectedGPU, model_name]
                # put cache in 
                evictModel = self.localLRU[selectedGPU].put(model_name, model_name)
                evict = []
                if evictModel:
                    evict = [selectedGPU, evictModel]
                self.updateCache(evict=evict, launch=launch)
                return selectedGPU, evict, launch
            
            # step 3 if no idle GPU as well then return None
            return None

    def getPlan(self, model_name, timeout=0.2):
        """ 
        try found one else wait for 10ms
        return gpu-id, evict[gpu-id, model-name], launch[gpu-id, model-name]
        """
        startT = time.time()
        found = None
        
        while found is None:
            self.reqNum += 1
            found = self._tryPlan(model_name)
            time.sleep(10/1e3) # 10ms
            # if time.time() - startT > timeout:
            #     raise Exception('time out for finding a plan')
        logging.debug("cache hit rate %f (%d/%d)", self.cacheHit / self.reqNum, self.cacheHit, self.reqNum)
        return found


class LoadBalancer:

    def __init__(self, cliPort, serverPort, server_map, fillWithTraining=False):
        """"""
        self.server_map = server_map

        self.fillWithTraining = fillWithTraining

        # after dispatched to gpu server
        self.processed_requests_mtx = thd.Lock()
        self.processed_requests = dict()
        self.incoming_requests = queue.Queue()  # from clients

        self.loads = dict()  # [gpu-id, task-count]
        self.loads_mtx = thd.Lock()
        self.workers = dict()  # [gpu-id, tcp-conn]
        self.workers_mtx = thd.Lock()
        self.worker_conn_mtxs = dict()

        self.trainingStatus = dict()  # [gpu-id, int]
        self.trainingStatus_mtx = thd.Lock()
        self.instance = dict()
        self.gpu_belong = dict()

        self.gpu_LRUCache = dict()  # [gpu-id, lru.LRUCache]
        self.gpu_LRUCache_mtx = thd.Lock()
        self.cache_loc = dict() # {}
        self.cache_loc_mtx = thd.Lock()
        
        self.request_id = 0
        self.request_id_mtx = thd.Lock()

        # client tcp server
        tcp_server = tcp.TcpServer('0.0.0.0', cliPort)
        server_lock = thd.Lock()
        for _ in range(8): # process pool for tcp acceptance
            cli_tcp_thd = thd.Thread(target=self._towardsClient, 
                                        args=(tcp_server, server_lock, self.incoming_requests))
            cli_tcp_thd.start()

        # server tcp server
        serv_tcp_proc = thd.Thread(
            target=self._towardsGPUServerThd, args=(serverPort,))
        serv_tcp_proc.start()

    def run(self,):
        """ the main function loop
        handle request from client
        dispatch it to GPU servers
        """
        n_proc = 8
        if self.fillWithTraining:
            """ start _fiilTrainingThd"""
            logging.debug("starting fill with training")
            _trainProc = thd.Thread(target=self._fillTrainingThd)
            _trainProc.start()
            logging.debug('fill Training started proc')

        request_consumers = []
        for _ in range(n_proc):
            p = thd.Thread(target=self._reqHandleThd)
            p.start()
            request_consumers.append(p)
        
        for p in request_consumers:
            p.join()

    def _towardsClient(self, server, server_lock, request_q):
        """ launch as a Process """

        while True:
            server_lock.acquire()
            conn = server.tcpAccept()
            server_lock.release()
            try:
                model_name = conn.tcpRecvWithLength()
                data = conn.tcpRecvWithLength()
                request_q.put([model_name, data, conn])
                logging.debug('client request model %s', model_name)
            except:
                logging.debug('failed receiving data from client')
                del conn


    def _towardsGPUServerThd(self, port):
        """ it need to modify the LB status
        accept connections from gpu servers
        receive GPU ids
        save connection obj
        """
        server = tcp.TcpServer('0.0.0.0', port)
        logging.info('tcp server for GPU servers started')
        while True:
            conn = server.tcpAccept()
            conn_mtx = thd.Lock()
            # register gpu ids
            gpus = conn.tcpRecvWithLength()
            self._registerGPUs(gpus, conn)

            # start threads to monitor response
            for _ in range(4):
                _p = thd.Thread(target=self._respHandleThd,
                                    args=(conn, conn_mtx))
                _p.start()

    def _registerGPUs(self, gpus, conn):
        gpus = gpus.decode().split(';')
        conn_lck = thd.Lock()
        inst_idx = len(self.instance)
        self.instance[inst_idx] = {'gpus': gpus, 'training': False}
        for g in gpus:
            self.gpu_belong[g] = inst_idx
            with self.workers_mtx:
                self.workers[g] = conn
                self.worker_conn_mtxs[g] = conn_lck

            with self.loads_mtx:
                self.loads[g] = 0
            # assume 10 client process
            self.gpu_LRUCache[g] = lru.LRUCache(12)
            logging.info('registered gpu %s', g)

            # if self.fillWithTraining
            # send start training proc message
            # init self.trainingStatus for each gpu
            # self.trainingStatus will be marked to 1 by fillTrainingThd
            if self.fillWithTraining:
                conn.tcpSendWithLength(b'create_train_proc')
                # conn.tcpSendWithLength(g.encode())

                # self.trainingStatus[g] = 0
                # logging.debug('training stats keys %s', self.trainingStatus.keys())

    def _respHandleThd(self, conn: tcp.TcpAgent, conn_mtx):  # pylint: disable=no-member
        """ read response from GPU server in thread
        and put it in res_q
        """
        while True:
            with conn_mtx:
                req_id = int(conn.tcpRecvWithLength().decode())
                cache_id = conn.tcpRecvWithLength().decode()  # gpu-id+model-name
                outputs = conn.tcpRecvWithLength()
            # reduce worker load
            gpu_id, model_name = cache_id.split('$$')
            with self.loads_mtx:
                self.loads[gpu_id] -= 1
            # reduce cache queue 
            with self.cache_loc_mtx:
                self.cache_loc[model_name][gpu_id] -= 1
            logging.debug('>>>>> resp, gpu %s, model %s', gpu_id, model_name)

            # response to client directly
            with self.processed_requests_mtx:
                cli_conn = self.processed_requests[req_id]
                try:
                    cli_conn.tcpSendWithLength(outputs)
                except Exception as e:
                    logging.info('client lost due to %s', str(e))
            del self.processed_requests[req_id]
            del cli_conn
            logging.debug("reponsed %s", req_id)

    def _reqHandleThd(self, ):
        server_map = self.server_map
        sch = Scheduler(self.loads, self.loads_mtx, self.gpu_LRUCache, 
                        self.gpu_LRUCache_mtx, self.cache_loc, self.cache_loc_mtx, server_map)
        ts_req = []
        while True:
            t1 = time.time()
            req = self.incoming_requests.get()
            # make sure while processing, training monitor will not send start training
            
            model_name = req[0].decode()
            t1 = time.time()
            # schedule it
            logging.debug("handle one incoming request %s", model_name)
            try:
                gpu_id, evict, launch = sch.getPlan(model_name, timeout=0.1)
            except Exception as e:
                logging.debug("handle schedule request for %s error try later. (err: %s)", model_name, str(e))
                self.incoming_requests.put(req)
                continue
            
            if evict:
                logging.debug(
                    '>>>>> evict, gpu %s, model %s', evict[0], evict[1])
                with self.workers_mtx:
                    conn = self.workers[evict[0]]
                
                with self.worker_conn_mtxs[evict[0]]:
                    conn.tcpSendWithLength(b'evict')
                    conn.tcpSendWithLength(
                        "{}$${}".format(evict[0], evict[1]).encode())
            if launch:
                logging.debug(
                    '>>>>> launch, gpu %s, model %s', launch[0], launch[1])
                with self.workers_mtx:
                    conn = self.workers[launch[0]]
                with self.worker_conn_mtxs[launch[0]]:
                    conn.tcpSendWithLength(b'create')
                    conn.tcpSendWithLength(launch[0].encode())
                    conn.tcpSendWithLength(launch[1].encode())
            
            with self.workers_mtx:
                conn = self.workers[gpu_id]

            with self.request_id_mtx:
                self.request_id += 1
                request_id = self.request_id

            with self.worker_conn_mtxs[gpu_id]:
                conn.tcpSendWithLength(b'inf')
                conn.tcpSendWithLength(str(request_id).encode())
                conn.tcpSendWithLength(
                    "{}$${}".format(gpu_id, model_name).encode())
                conn.tcpSendWithLength(req[1])
            logging.debug('>>>>> inf, gpu %s, model %s', gpu_id, model_name)
            # maintain tcp connection for response
            with self.processed_requests_mtx:
                self.processed_requests[request_id] = req[2]
            with self.loads_mtx:
                self.loads[gpu_id] += 1

            ts_req.append(time.time() - t1)
            if (len(ts_req) % 100 == 0):
                logging.info('avg request handle cost %s s', np.mean(ts_req[-100:]))
            logging.debug(
                'dispatched task to gpu id %s, request id %s', gpu_id, request_id)
    
    def _fillTrainingThd(self,):
        """ check self.incoming_requests queue, 
        if it is empty and some self.loads[gpu-id] is 0
            start training on that gpu
        """
  
        # server_map = controller_agent.listenController(controller_address, controller_port, lambda model_name: 'train' not in model_name)
        server_map = self.server_map
        
        logging.info("_fillTrainingProc")
        logging.debug('monite for necessary training')

        while True:
            inf_gpu_idxs = server_map.valid_server_list()
            with self.loads_mtx:
                GPU_ids = list(self.loads.keys())

            inf_GPUs = []
            for i in inf_gpu_idxs:
                if i < len(GPU_ids):
                    inf_GPUs.append(GPU_ids[i])

            train_GPUs = list(set(GPU_ids) - set(inf_GPUs))
            # logging.debug('gpus for inference size %s', len(inf_GPUs))
            # map selected gpus to instance ids
            instances = list([len(self.instance[i]['gpus']) for i in range(len(self.instance))])
            need_training_signal = []
            for gpu_id in train_GPUs:
                inst_idx = self.gpu_belong[gpu_id]
                instances[inst_idx] -= 1

                if instances[inst_idx] == 0:
                    need_training_signal.append(inst_idx)
            
            logging.debug("need_training_signal %s", str(need_training_signal))
            for idx in self.instance:
                if idx in need_training_signal:
                    if not self.instance[idx]['training']:
                        _gpu_id = self.instance[idx]['gpus'][0] # any gpu is fine
                        conn = self.workers[_gpu_id]
                        with self.worker_conn_mtxs[_gpu_id]:
                            conn.tcpSendWithLength(b'start_training')
                            self.instance[idx]['training'] = True
                        logging.debug('sending start training to instance %s', idx)
                else:
                    if self.instance[idx]['training']:
                        _gpu_id = self.instance[idx]['gpus'][0] # any gpu is fine
                        conn = self.workers[_gpu_id]
                        with self.worker_conn_mtxs[gpu_id]:
                            conn.tcpSendWithLength(b'stop_training')
                            self.instance[idx]['training'] = False
                        logging.debug('sending stop training to instance %s', idx)
            
            time.sleep(0.1)


def main():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, required=True, help="Address of the controller")
    parser.add_argument("--controller-port", type=int, required=True, help="Port of the controller")
    parser.add_argument("--client-port", type=int, required=True, help="Port for the client")
    parser.add_argument("--server-port", type=int, required=True, help="Port for the server")
    parser.add_argument("--fill-train", action='store_true', help="Flag to fill training data")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")

    args = parser.parse_args()

    controller_address = args.controller_address
    controller_port = args.controller_port
    client_port = args.client_port
    server_port = args.server_port

    if args.debug:
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO)

    server_map = controller_agent.listenController(controller_address, controller_port, lambda model_name: 'train' not in model_name)

    # lb = LoadBalancer(8777, 8778, server_map, args.fill_train)
    lb = LoadBalancer(client_port, server_port, server_map, args.fill_train)
    lb.run()


if __name__ == "__main__":
    main()
