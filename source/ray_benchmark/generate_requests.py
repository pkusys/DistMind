import requests
import threading
import numpy as np
from enum import Enum
import time
import argparse
import signal
import sys

from tcp import TcpClient
from ssh_comm import get_host_ips_slots

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", required=True)
    parser.add_argument("--output-stats", required=True, help="output the throughput statistic")
    parser.add_argument("--output-req-log", default="requests_log.txt")
    parser.add_argument("--interval", default=1, type=int, help="statistic interval in seconds")
    parser.add_argument("--controller-ip", default="127.0.0.1", help="controller ip")
    parser.add_argument("--controller-port", default=8014, type=int, help="controller port")
    parser.add_argument("--server-port", default=8000, type=int, help="server port")
    parser.add_argument("--max-requests", default=0, type=int, help="maximum number of requests to generate (0 = unlimited)")
    return parser.parse_args()

def query_model_ready(ip, port, gpu_id):
    url=f"http://{ip}:{port}/gpu-{gpu_id}/model_ready"
    resp = requests.get(url)
    while resp.status_code != 200:
        resp = requests.get(url)
    return resp.json()['ready']

def inference_query(ip, port, gpu_id, model_idx):
    query_url = f"http://{ip}:{port}/gpu-{gpu_id}/inference/{model_idx}"
    resp = requests.get(query_url)
    return resp

def query_inference_count(ip, port, gpu_id, results):
    url=f"http://{ip}:{port}/gpu-{gpu_id}/inference_count"
    resp = requests.get(url)
    while resp.status_code != 200:
        resp = requests.get(url)

    key = f'{ip}:{gpu_id}'
    results[key] = resp.json()['count']

def query_training_count(ip, port, gpu_id, results):
    url=f"http://{ip}:{port}/gpu-{gpu_id}/training_batch_count"
    resp = requests.get(url)
    while resp.status_code != 200:
        resp = requests.get(url)

    key = f'{ip}:{gpu_id}'
    results[key] = resp.json()['count']

class Workload(Enum):
    NULL = 0
    INFERENCE = 1
    TRAINING = 2

class WorkloadGenerator:
    def __init__(self, ip, port, gpu_id, log_file=None, max_requests=0):
        """"""
        self.log_file = log_file
        if self.log_file is not None:
            self.log_file = open(self.log_file, 'w')

        self.shutdown = False
        self.inf_bs = 8
        self.node_ip = ip
        self.node_port = port
        self.gpu_id = gpu_id

        self.query_data = np.random.rand(self.inf_bs, 3, 224, 224).astype(np.float32).tobytes()
        self.workload : Workload = Workload.NULL
        self.gpu_is_training = False

        self.inference_model_id = -1
        self.last_inference_model_id = -1
        self.workload_lock = threading.RLock()

        self.max_requests = max_requests
        self.completed_requests = 0
        self.request_complete_event = threading.Event()

        self.daemon_thd = threading.Thread(target=self._background, args=())
        self.daemon_thd.daemon = False  # Don't make thread daemonic to ensure proper cleanup
        self.daemon_thd.start()

        self.inference_history = []
    def log(self, msg):
        out_msg = f"GPU {self.node_ip}:{self.gpu_id} => {msg}"
        if self.log_file is None:
            print(out_msg)
        else:
            self.log_file.write(f"{out_msg}\n")

    def _background(self,):
        while not self.shutdown:
            """"""
            if self.workload == Workload.INFERENCE:
                """make inference request """
                self.log(f'workload {self.workload}')
                with self.workload_lock:
                    model_id = self.inference_model_id
                    # Check for shutdown again after acquiring lock
                    if self.shutdown:
                        break
                        
                inf_start = time.time()
                self.log(f'query inf {model_id}')
                
                try:
                    resp = inference_query(self.node_ip, self.node_port, 
                                          self.gpu_id, model_id)
                    inf_end = time.time()
                    
                    # Check for shutdown again before processing response
                    if self.shutdown:
                        break
                        
                    self.log(resp.json())
                    self.inference_history.append(
                        [inf_start, inf_end, model_id]
                    )
                    self.log(f'Inference takes {(inf_end - inf_start) * 1e3}ms')
                      # Increment completed requests count and check if we've reached the maximum
                    with self.workload_lock:
                        self.completed_requests += 1
                        current_completed = self.completed_requests
                        if self.max_requests > 0 and current_completed >= self.max_requests:
                            old_workload = self.workload
                            self.workload = Workload.NULL
                            self.log(f"Worker completed its {self.max_requests} max_requests, changing workload from {old_workload} to NULL")
                            self.request_complete_event.set()
                except Exception as e:
                    if not self.shutdown:  # Only log if not shutting down
                        self.log(f"Inference request failed: {e}")
                
                self.last_inference_model_id = model_id

            elif self.workload == Workload.TRAINING:
                """ launch training """
                # Check for shutdown before launching training
                if self.shutdown:
                    break
                    
                self.launch_training()
                self.last_inference_model_id = -1
                  # Count training as a completed request
                with self.workload_lock:
                    self.completed_requests += 1
                    current_completed = self.completed_requests
                    if self.max_requests > 0 and current_completed >= self.max_requests:
                        old_workload = self.workload
                        self.workload = Workload.NULL
                        self.log(f"Worker completed its {self.max_requests} max_requests, changing workload from {old_workload} to NULL")
                        self.request_complete_event.set()
            else:
                # Use shorter sleep times to respond to shutdown requests more quickly
                time.sleep(0.001)
    
    
    def launch_training(self):
        if self.gpu_is_training:
            return
        resp = requests.get(url=f"http://{self.node_ip}:{self.node_port}/gpu-{self.gpu_id}/train")
        if resp.status_code == 200 and resp.json()['result'] == "ok":
            self.log("start training successfully")
            self.gpu_is_training = True  
        else:
            self.log(f'start training failed resp {resp}')
    
    def set_workload(self, workload: Workload, inference_model_id=-1,):
        with self.workload_lock:
            self.workload = workload
            if self.workload == Workload.INFERENCE:
                assert inference_model_id != -1
                self.inference_model_id = inference_model_id
                print(f'set inference model id {self.inference_model_id}')
    
    def wait_for_completion(self, timeout=None):
        """Wait until all requested operations are completed"""
        return self.request_complete_event.wait(timeout) 

    def is_completed(self):
        """Check if all requested operations are completed"""
        if self.max_requests <= 0:
            return False
        with self.workload_lock:
            is_done = self.completed_requests >= self.max_requests
            if is_done and not hasattr(self, '_completion_logged'):
                self.log(f"Worker has completed its max_requests: {self.completed_requests}/{self.max_requests}")
                # Add a flag to avoid repeated logging
                self._completion_logged = True
            return is_done
    
    def cleanup(self):
        """Clean up resources before exit"""
        self.shutdown = True
        if self.daemon_thd.is_alive():
            self.daemon_thd.join(timeout=2.0)  # Wait for thread to exit with timeout
        if self.log_file is not None:
            try:
                self.log_file.flush()
                self.log_file.close()
            except:
                pass
    
def stats_thread_fn(interval, output_file, node_ips, ngpus, port, stop_event):
    last_inference_counts = {}
    last_training_counts = {}

    log_file = open(output_file, 'w')

    while not stop_event.is_set():
        t_start = time.time()
        cur_inference_counts = {}
        cur_training_counts = {}
        query_threads = []
        for j in range(len(node_ips)):
            ip = node_ips[j]
            for i in range(ngpus[j]):
                inf_t = threading.Thread(
                    target=query_inference_count, 
                    args=(ip, port, i, cur_inference_counts))
                inf_t.start()
                train_t = threading.Thread(
                    target=query_training_count,
                    args=(ip, port, i, cur_training_counts))
                train_t.start()
                query_threads.append(inf_t)
                query_threads.append(train_t)

        for t in query_threads:
            t.join()
        # compute the difference
        if len(last_inference_counts) > 0:
            throughputs = []
            for key in cur_training_counts:
                cur_inf = cur_inference_counts[key]
                last_inf = last_inference_counts[key]
                cur_train = cur_training_counts[key]
                last_train = last_training_counts[key]
                print(f"key {key} cur_inf {cur_inf} last_inf {last_inf} cur_train {cur_train} last_train {last_train}")

                d_inf = cur_inf - last_inf
                d_train = cur_train - last_train

                throughputs.append(d_inf)
                throughputs.append(d_train)
            # write out
            output_str = ", ".join([str(v) for v in throughputs])
            print(f'throughput:: {output_str}')
            log_file.write(f"{time.time()}, {output_str}\n")
            log_file.flush()
        
        last_inference_counts = cur_inference_counts
        last_training_counts = cur_training_counts
        t_end = time.time() 
        pause_time = interval - (t_end - t_start)
        if pause_time > 0:
            # Use wait with timeout to allow early exit
            if stop_event.wait(pause_time):
                break
    
    log_file.close()
    print("Stats thread exiting")

def main():
    """"""
    args = get_args()
    controller_ip = args.controller_ip
    controller_port = args.controller_port
    max_requests = args.max_requests

    server_port = args.server_port
    node_ips, slots = get_host_ips_slots(args.hostfile)

    for i in range(len(node_ips)):
        ip = node_ips[i]
        for j in range(slots[i]):
            # Check if the model is ready
            while not query_model_ready(ip, server_port, j):
                print(f"Waiting for model to be ready on {ip}:{j}...")
                time.sleep(10)

    # Create an event to signal when all work is done
    stop_event = threading.Event()    # start the stats gathering thread
    stats_thd = threading.Thread(
                    target=stats_thread_fn, 
                    args=(args.interval, args.output_stats, node_ips, slots, server_port, stop_event))
    stats_thd.daemon = False  # Don't make thread daemonic to ensure proper cleanup
    stats_thd.start()

    req_generators = {}
    for i in range(len(node_ips)): 
        for j in range(slots[i]):
            _id = f"{node_ips[i]}:{j}"
            req_generators[_id] = WorkloadGenerator(
                node_ips[i], server_port, j, log_file=f"{args.output_req_log}-{_id}", max_requests=max_requests)

    # connect to controller to get the workload
    controller_cli = TcpClient(controller_ip, controller_port)    # Helper function to check if all workers have completed their individual max_requests
    def all_workers_completed():
        if max_requests <= 0:
            return False
        
        completed_workers = 0
        total_workers = len(req_generators)
        for server_id, generator in req_generators.items():
            if generator.is_completed():
                completed_workers += 1
        
        if completed_workers > 0:
            print(f"Worker completion status: {completed_workers}/{total_workers} workers completed")
        
        return completed_workers == total_workers and total_workers > 0
        
    request_count = 0
    try:
        while True:
            # Check if all workers have completed their max_requests
            if max_requests > 0 and all_workers_completed():
                print(f"All workers have completed their individual max_requests of {max_requests}, exiting main loop")
                break

            server_id = controller_cli.tcpRecvWithLength().decode("utf-8")
            model_name:str = controller_cli.tcpRecvWithLength().decode("utf-8")
            controller_cli.tcpSend(b"RECV")
            print(f'{time.time()}: {server_id} {model_name}')

            g: WorkloadGenerator = req_generators[server_id]
            if "alter" in model_name:
                id_ = int(model_name.rsplit("-", 1)[-1])
                print(f'changing workload to {model_name}')
                g.set_workload(Workload.INFERENCE, id_)
                request_count += 1
                
                # Print status every 5 requests
                if request_count % 5 == 0:
                    print(f"\nProcessed {request_count} requests. Worker status:")
                    total_completed = 0
                    for sid, gen in req_generators.items():
                        with gen.workload_lock:
                            completed = gen.completed_requests
                            total_completed += completed
                        print(f"  {sid}: {completed}/{max_requests} completed")
                    
                    # Check if all workers have completed
                    if max_requests > 0 and all_workers_completed():
                        print(f"All workers have completed their individual max_requests of {max_requests}, exiting main loop")
                        break
                        
            elif "train" in model_name:
                id_ = -1
                g.set_workload(Workload.TRAINING, id_)
                request_count += 1
            else:
                print(f"unknown workload for {server_id}: {model_name}")        # Wait for all generators to complete their requests
        if max_requests > 0:
            print(f"Waiting for all {len(req_generators)} workers to complete {max_requests} requests each...")
            
            all_completed = False
            check_interval = 0.1  # seconds between checks
            status_report_interval = 10  # Report status every X checks
            check_count = 0
            
            while not all_completed and not stop_event.is_set():
                all_completed = True
                completed_workers = 0
                
                # Check each worker's completion status
                for server_id, generator in req_generators.items():
                    if generator.is_completed():
                        completed_workers += 1
                    else:
                        all_completed = False
                
                # Report status periodically
                if check_count % status_report_interval == 0:
                    print(f"Progress: {completed_workers}/{len(req_generators)} workers completed")
                    # Print status of incomplete workers
                    if not all_completed:
                        print("Incomplete workers:")
                        for server_id, gen in req_generators.items():
                            if not gen.is_completed():
                                with gen.workload_lock:
                                    print(f"  {server_id}: {gen.completed_requests}/{max_requests} completed, workload={gen.workload}")
                
                if not all_completed:
                    time.sleep(check_interval)
                    check_count += 1
            
            if all_completed:
                print("All workers successfully completed their individual max_requests!")
            elif stop_event.is_set():
                print("Shutdown requested before completion")
    finally:
        # Signal threads to stop and clean up
        stop_event.set()
        
        # Properly clean up each generator
        for gen in req_generators.values():
            gen.cleanup()
        
        # Wait for stats thread to complete
        if stats_thd.is_alive():
            stats_thd.join(timeout=5.0)
        
        print("Request generator shutting down")

if __name__ == "__main__":
    main()