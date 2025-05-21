import sys
import struct
import socket
import pickle

import numpy
import torch
import deployment_c
from model.common.util import extract_func_info
from model.index import get_model_module
from py_utils.tcp import TcpClient

KVSTORAGE_OP_READ = 0
KVSTORAGE_OP_WRITE = 1
KVSTORAGE_OP_ACK = 2

def import_storage_list(filename):
    storage_list = []
    with open(filename) as f:
        _ = f.readline()
        for line in f.readlines():
            parts = line.split(',')
            storage_address = parts[0].strip()
            storage_port = int(parts[1].strip())
            storage_list.append((storage_address, storage_port))
    return storage_list

def import_model_distribution_map(filename):
    model_distribution_map = {}
    with open(filename) as f:
        _ = f.readline()
        for line in f.readlines():
            parts = line.split(',')
            model_name = parts[0].strip()
            distributed = [int(part.strip()) for part in parts[1:]]
            model_distribution_map[model_name] = distributed
    return model_distribution_map

def write_to_metadata_storage(hostname, key, value_b):
    op = KVSTORAGE_OP_WRITE
    op_b = struct.pack('I', op)
    metadata_client = TcpClient(hostname[0], hostname[1])
    metadata_client.tcpSend(op_b)
    metadata_client.tcpSendWithLength(key.encode())
    metadata_client.tcpSendWithLength(value_b)
    ret_b = metadata_client.tcpRecvWithLength()
    print (ret_b.decode())
    del metadata_client

def main():
    storage_filename = sys.argv[1]
    model_filename = sys.argv[2]
    binary_filename = sys.argv[3]
    shm_name = sys.argv[4]
    shm_size = int(sys.argv[5])

    # Initialize shared memory in cpp_extension
    deployment_c.initialize(shm_name, shm_size)

    # Import storage list
    storage_list = import_storage_list(storage_filename)
    efa_storage_list = []
    for addr, port in storage_list[1:]:
        store_cli_name = 'deployment_store_client_%d' % len(efa_storage_list)
        store_cli_id_b = socket.inet_aton(addr) + struct.pack('I', port)
        store_cli_id = struct.unpack('Q', store_cli_id_b)[0]
        deployment_c.connect(store_cli_name, addr, port)
        efa_storage_list.append((store_cli_name, store_cli_id))

    # Import model list
    model_distribution_map = import_model_distribution_map(model_filename)
    
    # Deploy models
    with open(binary_filename, 'rb') as f:
        num_models_b = f.read(4)
        print (num_models_b)
        num_models = struct.unpack('I', num_models_b)[0]
        for _ in range(num_models):
            model_name_length_b = f.read(8)
            model_name_length = struct.unpack('Q', model_name_length_b)[0]
            model_name_b = f.read(model_name_length)
            model_name = model_name_b.decode()
            model_size_b = f.read(8)
            model_size = struct.unpack('Q', model_size_b)[0]
            num_batches_b = f.read(4)
            num_batches = struct.unpack('I', num_batches_b)[0]

            distributed = model_distribution_map[model_name]
            for _ in range(num_batches):
                kv_size_b = f.read(8)
                # kv_size = struct.unpack('Q', kv_size_b)[0]
                key_size_b = f.read(8)
                key_size = struct.unpack('Q', key_size_b)[0]
                key_b = f.read(key_size)
                key = key_b.decode()
                value_size_b = f.read(8)
                value_size = struct.unpack('Q', value_size_b)[0]
                value_b = f.read(value_size)

                average_size = value_size // len(distributed) + 1
                batch_location = []
                for storage_id, storage in enumerate(distributed):
                    slice_offset = average_size * storage_id
                    slice_ending = min(average_size * (storage_id + 1), value_size)
                    slice_size = slice_ending - slice_offset
                    slice_name = key + ('-SLICE-OFFSET-%d' % slice_offset) + ('-SLICE-SIZE-%d' % slice_size)
                    slice_b = value_b[slice_offset: slice_ending]

                    store_cli_name, store_cli_id = efa_storage_list[storage]
                    deployment_c.put_kv_bytes(store_cli_name, slice_name, slice_b, slice_size)
                    batch_location.append((slice_offset, slice_size, store_cli_id))
                    print ('\t', store_cli_name, slice_name)
                # Insert batch metadata to metadata_storage
                batch_location_key = key + '-LOCATION'
                batch_location_b = b''
                for slice_offset_b, slice_size_b, store_cli_id in batch_location:
                    batch_location_b += struct.pack('QQQ', slice_offset_b, slice_size_b, store_cli_id)
                write_to_metadata_storage(storage_list[0], batch_location_key, batch_location_b)
                print ('\t', batch_location_key, len(batch_location_b))
            
            # Insert model metadata to storage
            kv_size_b = f.read(8)
            # kv_size = struct.unpack('Q', kv_size_b)[0]
            key_size_b = f.read(8)
            key_size = struct.unpack('Q', key_size_b)[0]
            key_b = f.read(key_size)
            key = key_b.decode()
            value_size_b = f.read(8)
            value_size = struct.unpack('Q', value_size_b)[0]
            value_b = f.read(value_size)

            store_cli_name, store_cli_id = efa_storage_list[distributed[0]]
            model_metadata_slice_key = key + ('-SLICE-OFFSET-%d' % 0) + ('-SLICE-SIZE-%d' % len(value_b))
            deployment_c.put_kv_bytes(store_cli_name, model_metadata_slice_key, value_b, len(value_b))
            print (model_metadata_slice_key, len(value_b), store_cli_id)

            # Insert model metadata location to metadata storage
            model_metadata_location_key = key + '-LOCATION'
            model_metadata_location_b = struct.pack('QQQ', 0, len(value_b), store_cli_id)
            write_to_metadata_storage(storage_list[0], model_metadata_location_key, model_metadata_location_b)
            print (model_metadata_location_key, len(model_metadata_location_b))

            # Insert model size to metadata_storage
            model_size_key = model_name + "-SIZE"
            model_size_b = struct.pack('Q', model_size)
            write_to_metadata_storage(storage_list[0], model_size_key, model_size_b)
            print (model_size_key, len(model_size_b))

if __name__ == "__main__":
    main()