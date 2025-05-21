import posix_ipc
import sys
from model.index import get_model_module
import pickle
import logging
import mmap
import subprocess
from multiprocessing import Pool

logging.basicConfig(
    format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG)


def save_shm(model_name):
    """"""
    # fake for now
    model_module = get_model_module(model_name)
    model = model_module.import_model()
    model.fullname = model_name
    # model.share_memory()
    model_b = pickle.dumps(model)
    _size = len(model_b)
    
    # open shm 
    memory = posix_ipc.SharedMemory(model_name, posix_ipc.O_CREX, size=_size)
    memfile = mmap.mmap(memory.fd, memory.size)
    memory.close_fd()
    # write model bytes
    memfile.seek(0)
    memfile.write(model_b)
    memfile.close()
    logging.info('loaded model %s, size %s', model_name, _size)
    return [model_name, _size]


def main():
    """"""
    if len(sys.argv) < 3:
        print('require a model list to load, and output file for saving model sizes')
        return

    subprocess.run('rm -rf /dev/shm/* 2>/dev/null', shell=True)

    model_names = []
    with open(sys.argv[1]) as ifile:
        _ = ifile.readline() # ignore first line
        for line in ifile:
            # model_name, param_path = line.strip('\n').split(',')
            name = line.strip('\n')
            model_names.append(name)
    
    with Pool(4) as p:
        model_sizes = p.map(save_shm, model_names)

    logging.info('all models loaded')
    logging.info('writing out model sizes')
    with open(sys.argv[2], 'w') as ofile:
        for m, s in model_sizes:
            ofile.write("{}, {}\n".format(m, s))


if __name__ == "__main__":
    main()
