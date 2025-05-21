"""
update environment variable to 
launch act as MPS client
accept tasks through task_q, and send response back through 
response queue


"""
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import logging
import pickle
import posix_ipc
import mmap
import time
from model.index import get_model_module
import torch.distributed as dist


input_shapes = {
    'resnet152': [-1, 3, 224, 224],
    'inception_v3': [-1, 3, 299, 299],
    'densenet201': [-1, 3, 224, 224],
    'bert_base': [-1, 512],
    'gpt2': [-1, 512]
}


def get_inputshape(model_name):
    prefix = model_name.split('-')[0]
    return input_shapes[prefix]


def workerProc(gpu_id, model_name, model_size, is_inf,
               task_q: mp.Queue, res_q: mp.Queue,
               env, log_dir, one_step_train_fn=None):
    """"""
    base_mps_dir = '/tmp/nvidia-mps/'
    base_log_dir = '/tmp/nvidia-log/'

    if log_dir is not None:
        here = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.abspath(os.path.join(here, log_dir))
        if not os.path.exists(log_folder):
            try:
                os.makedirs(log_folder)
            except Exception as e:
                print(e)
        log_filename = os.path.join(log_folder, "{}.inf.log".format(gpu_id))
        logfile = open(log_filename, "a")

    logging.debug('enter worker proc')
    # make sure launch task on given gpu id
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    env['CUDA_MPS_PIPE_DIRECTORY'] = base_mps_dir + gpu_id
    env['CUDA_MPS_LOG_DIRECTORY'] = base_log_dir + gpu_id
    os.environ.update(env)
    device = 'cuda'

    # load model from shm
    t1 = time.time()
    first = True
    memory = posix_ipc.SharedMemory(model_name, size=model_size)
    memfile = mmap.mmap(memory.fd, memory.size)
    memory.close_fd()
    memfile.seek(0)
    cpu_model = pickle.load(memfile)
    memfile.close()
    logging.debug('shm unpickle takes %s s', time.time() - t1)

    model_name = cpu_model.fullname
    input_shape = get_inputshape(model_name)
    logging.debug('before model move to cuda')
    model = cpu_model.to(device)
    model.eval()
    logging.debug('model launched on gpu %s, inf mode %s', gpu_id, is_inf)
    logging.debug('model is training, %s', model.training)
    del cpu_model
    del memfile
    del memory
    if is_inf:
        while True:
            t = task_q.get()  # first part as GPUid+model-name
            if not first:
                t1 = time.time()
            first = False
            if t == "exit":
                logging.debug('received exit signal')
                break
            data = t[1]
            # pylint: disable=no-member
            if 'bert_base' in model_name or 'gpt2' in model_name:
                data = torch.from_numpy(np.frombuffer(data, dtype=np.int64))
            else:
                data = torch.from_numpy(np.frombuffer(data, dtype=np.float32))
            data = data.to(device).view(input_shape)
            outputs = model(data)
            if 'bert_base' in model_name or 'gpt2' in model_name:
                ob = outputs[0].detach().cpu().numpy().tobytes()
            else:
                ob = outputs.detach().cpu().numpy().tobytes()
            res_q.put([t[0], ob])
            
            if log_dir is not None:
                logfile.write("inference time: {}\n".format(time.time() - t1))
                logfile.flush()
            logging.debug('completed inference task %s', t[0])
    else:
        # training case
        while True:
            t = task_q.get(False)
            if t != None and t == 'exit':
                break
            elif t != 'exit':
                print('unsupported task')
                break
            else:
                # do one training iteraion
                one_step_train_fn(model)
    logging.debug('worker proc of model %s ends', model_name)


def trainingProc(gpu_id, env, control_val: mp.Value, local_rank, world_size, train_log_dir=None):
    """  
    """
    base_mps_dir = '/tmp/nvidia-mps/'
    base_log_dir = '/tmp/nvidia-log/'
    dist_url = "tcp://127.0.0.1:23451"
    
    here = os.path.dirname(os.path.abspath(__file__))
    if train_log_dir is not None:
        log_folder = os.path.abspath(os.path.join(here, train_log_dir))
    else:
        log_folder = os.path.abspath(os.path.join(here, '../../tmp/test3/mps/training_logs'))
    if not os.path.exists(log_folder):
        try:
            os.makedirs(log_folder)
        except Exception as e:
            print(e)
    log_filename = os.path.join(log_folder, "{}.train.log".format(gpu_id))
    logfile = open(log_filename, "w")

    logging.debug('enter training proc')
    # make sure launch task on given gpu id
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    env['CUDA_MPS_PIPE_DIRECTORY'] = base_mps_dir + gpu_id
    env['CUDA_MPS_LOG_DIRECTORY'] = base_log_dir + gpu_id
    os.environ.update(env)
    device = 'cuda'

    dist.init_process_group(backend='nccl', init_method=dist_url, rank=local_rank, world_size=world_size)
    # torch.cuda.set_device(local_rank)

    mod = get_model_module('resnet152')
    model = mod.import_model(train=True)
    model = model.to(device)
    # wrap model with distributed Data-parallel
    # because using MPS attached each process to single GPU, not need to specify GPU idx
    model = torch.nn.parallel.DistributedDataParallel(model) 
    model.fullname = "resnet152"

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    name = model.fullname
    data_fn = get_model_module(name).import_data
    bs = 16 # TODO: modify later
    data, _ = data_fn(bs)
    # target = torch.nn.functional.one_hot(target, 1000)
    
    data = data.to(device)
    t1 = time.time()
    
    while True:
        # condition logging
        if time.time() - t1 > 10:
            logging.debug("control_val.value %s", control_val.value)
            t1 = time.time()
        
        if control_val.value == 1:
            target = np.random.randint(0, 1000, size=bs)
            target = torch.tensor(target, device=device) # pylint: disable=not-callable
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.debug('training in progress:: loss: %s', loss.sum().item())
            logfile.write("{},1\n".format(time.time()))
            logfile.flush()
        else:
            # logging.debug('stop training')
            time.sleep(0.05)
    