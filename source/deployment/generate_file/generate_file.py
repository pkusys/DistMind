import sys
import struct
import socket
import pickle

import torch
import deploy_generate_c
from model.common.util import extract_func_info
from model.index import get_model_module
from py_utils.tcp import TcpClient

def import_model_list(filename):
    model_list = []
    with open(filename) as f:
        _ = f.readline()
        for line in f.readlines():
            parts = line.split(',')
            model_name = parts[0].strip()
            model_list.append(model_name)
    return model_list

def import_model(model_name, max_batch_size):
    module = get_model_module(model_name)

    data, target = module.import_data(8)
    shape = ((data.numpy().dtype, list(data.shape)[1:]), (target.numpy().dtype, list(target.shape)[1:]))
    
    func_list, batch_list = module.import_model_reimpl_with_batching(train=('train' in model_name), max_batch_size=max_batch_size)
    func_info, param_list = extract_func_info(func_list)

    def padd512(size):
        return (size + 511) // 512 * 512
    def get_param_size(params):
        return sum([padd512(p.nelement() * p.element_size()) for _, p in params.items()])
    model_size = sum([get_param_size(params) for params in param_list])

    return shape, func_info, param_list, batch_list, model_size

# def write_to_metadata_storage(hostname, key, value_b):
#     op = KVSTORAGE_OP_WRITE
#     op_b = struct.pack('I', op)
#     metadata_client = TcpClient(hostname[0], hostname[1])
#     metadata_client.tcpSend(op_b)
#     metadata_client.tcpSendWithLength(key.encode())
#     metadata_client.tcpSendWithLength(value_b)
#     ret_b = metadata_client.tcpRecvWithLength()
#     print (ret_b.decode())
#     del metadata_client

def main():
    model_filename = sys.argv[1]
    filename = sys.argv[2]
    # 0 for per layer, 512 * 1024 * 1024  for per app
    try:
        max_batch_size = int(sys.argv[3])
    except:
        max_batch_size = 8 * 4096000

    # Open the image file in cpp_extension
    deploy_generate_c.initialize(filename)

    # Import model list
    model_list = import_model_list(model_filename)
    
    # Deploy models
    for model_name in model_list:
        shape, func_info, param_list, batch_list, model_size = import_model(model_name, max_batch_size)
        deploy_generate_c.put_model_profile(model_name, model_size, len(batch_list) - 1)

        # Create batched parameters
        batch_info_list = []
        for batch_id in range(len(batch_list) - 1):
            batch_name = model_name + ('-BATCH-%d' % batch_id)
            
            start_layer = batch_list[batch_id][0]
            end_layer = batch_list[batch_id + 1][0]
            batch_total_size = batch_list[batch_id + 1][1]
            batch_total_numel = batch_total_size // 4

            batch_data = torch.zeros(batch_total_numel, dtype=torch.float32)
            offset = 0
            for params in param_list[start_layer: end_layer]:
                for _, p in params.items():
                    real_numel = p.nelement()
                    padded_numel = (real_numel + 127) // 128 * 128
                    batch_data[offset: offset + real_numel] = p.view(-1)
                    offset += padded_numel

            batch_info_list.append((batch_name, batch_total_size, batch_data))
            deploy_generate_c.put_kv_tensor(batch_name, batch_data)
            print (batch_name, batch_total_size, batch_data.sum().item())

        # Insert model metadata to storage
        model_metadata_key = model_name + '-METADATA'
        model_metadata_b = b''
        model_pyinfo = (shape, func_info, batch_list)
        model_pyinfo_b = pickle.dumps(model_pyinfo)
        model_pyinfo_length = len(model_pyinfo_b)
        model_pyinfo_length_b = struct.pack('I', model_pyinfo_length)
        model_metadata_b += model_pyinfo_length_b
        model_metadata_b += model_pyinfo_b
        num_batch = len(batch_info_list)
        num_batch_b = struct.pack('I', num_batch)
        model_metadata_b += num_batch_b
        for batch_name, _, _ in batch_info_list:
            batch_name_b = batch_name.encode()
            batch_name_length = len(batch_name)
            batch_name_length_b = struct.pack('I', batch_name_length)
            model_metadata_b += batch_name_length_b
            model_metadata_b += batch_name_b
        deploy_generate_c.put_kv_bytes(model_metadata_key, model_metadata_b, len(model_metadata_b))
        print (model_metadata_key, len(model_metadata_b))

    deploy_generate_c.finalize()

if __name__ == "__main__":
    main()