import os
import nvgpu
import subprocess


def get_gpus():
    """ get GPU ids
    """
    ids = []
    for item in nvgpu.gpu_info():
        ids.append(item['uuid'])
    return ids


def main():
    """"""
    gpu_ids = get_gpus()

    base_mps_dir = '/tmp/nvidia-mps/'
    base_log_dir = '/tmp/nvidia-log/'
    if not os.path.exists(base_mps_dir):
        os.makedirs(base_mps_dir)
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)
    
    for i in gpu_ids:
        envs = os.environ.copy()
        envs['CUDA_VISIBLE_DEVICES'] = i
        envs['CUDA_MPS_PIPE_DIRECTORY'] = base_mps_dir + i
        envs['CUDA_MPS_LOG_DIRECTORY'] = base_log_dir + i
        subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True, env=envs)

        print('shutdown mps daemon on gpu ', i)

if __name__ == "__main__":
    main()