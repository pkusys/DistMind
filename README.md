# DistMind

This is the artifact of the paper "DistMind: Efficient Resource Disaggregation for Deep Learning Workloads". We are going to guide you through the process of reproducing the main results in the paper.

Here is a high level overview of the whole process:

1. Environment setup: Create GPU and memory instances on AWS (or machines with rdma).
2. Kick-the-tires: Run an example to verify DistMind are working.
3. Run: Run experiments.

Note that all logs for tests will be stored under ./tmp and figures will be stored under ./AE/{testname}.

## Environment Setup

Originally, the experiments were run in AWS EC2 instances p3dn.24xlarge and c5n.18xlarge. However, Amazon has changed their rules, making these two instances no longer support RDMA. Based on present rules, we recommend g6.12xlarge for GPU servers and c6in.32xlarge for memory servers. We also provide verbs version if you run tests on machines using verbs API rdma.

### Prepare EFA

If you use AWS instances, follow the instructions on [AWS User Guide](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/efa.html) to setup efa and nccl. **You should run code under branch efa**.

### Prepare Verbs

If you use verbs API rdma, follow the instructions on [libfabric install guide](https://github.com/ofiwg/libfabric) to install libfabric to /opt/libfabric. You should install nccl if your machine supported. **You should run code under branch main**.

### Prepare env

1. make sure you have finished one of the previous section  
2. cuda-12.4 / cuDNN: 9.5.1  
3. anaconda
4. pybind11: `git submodule update --recursive --init`
5. spdlog: `sudo apt install libspdlog-dev`
6. libtorch: download from [Website](https://download.pytorch.org/libtorch/cu124/libtorch-shared-with-deps-2.5.1%2Bcu124.zip) and unzip to {proj_path}/libtorch

### Prepare python env

1. `conda create -n distmind python=3.10 matplotlib`, and add activation to ~/.bashrc
2. `pip install transformers==4.49`
3. `pip install ray[serve]==1.13`
4. `pip install nvgpu`
5. `pip install posix_ipc`
6. `pip install parallel-ssh`
7. `pip install pydantic==1.10.8`
8. `conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia`

## Build

In project dictionary run the following command:

```sh
mkdir build
cd build
cmake ..
make -j 8

cd ../source/server/cpp_extension/torch
python setup.py install
```

When building end successfully, and following to ~/.bashrc

```sh
# python path
proj_path="~/DistMindAE" # replace path with absolute path
export PYTHONPATH="$proj_path/build/lib/python:$proj_path/build/lib:$proj_path:$PYTHONPATH"
```

If you are using AWS instances for testing, save this as an AMI to create multiple machines.

## Kick-the-tires

Follow the steps below to run an example test.

1. prepare one gpu server, we will run an example test on single machine.
2. check ips in settings/config.sh are all 127.0.0.1, MODE=local, modify GPU_LIST and WORLD_SIZE based on your machine's resources. Make sure settings/storage_list.txt is something like

```sh
storage_address,        storage_port
127.0.0.1,            7777
127.0.0.1,            7778
```

3. In the project path, run the following cmd.

```sh
mkdir -p tmp
mkdir -p tmp/test1
mkdir -p tmp/test1/distmind_remote
./AE/1_Meeting_latency_SLOs/run_distmind_test1.sh
```

4. When the script end, check tmp/test1/distmind_remote/log_client.txt. If in the end it says "All threads finished.", you have started distmind successfully.

## Run

Before running any test, you should modify files in settings correctly. You should prepare 4 memory servers and 4 GPU servers ideally, but you can reduce the size if your resources are limited. Choose one memory server as local and modify the settings as instructed below:

1. settings/serverhost_list.txt: add your server ip in to each line with format "[ip] slots=[gpu_num]"
2. settings/storage_list.txt: replace the first line's ip with local ip, keep port as 7777. Then add your memory server with format "[ip],    7778"
3. settings/controller.json & settings/mps_controller.json & settings/ray_controller.json: fill inference_workload_s with server_number * gpu_per_server
4. settings/config.sh: replace all ips with local ip, set MODE=remote
5. in each gpu server: in settings/config.sh replace LOCAL_IP with the server's ip, modify GPU_LIST and WORLD_SIZE based on your machine's resources
6. you should enable password free ssh connection for all servers, and replace the username in settings/username.txt with your config

### Meeting latency SLOs

In local's terminal, run `./AE/1_Meeting_latency_SLOs/run_test1.sh`. When the script finished(without error), run `python ./AE/1_Meeting_latency_SLOs/drawplot.py`. The plot will be saved to  
./AE/1_Meeting_latency_SLOs/fig6.png

### End-to-end performance

In local's terminal, run `./AE/2_End-to-end_performance/run_test2.sh`. When the script finished(without error), run `python ./AE/2_End-to-end_performance/drawplot.py`. The plot will be saved to  
./AE/2_End-to-end_performance/fig7.png & ./AE/2_End-to-end_performance/fig8.png

### Sharing inference and training

In this test, you should modify the inference_workload_s list in three controller.json files. Assume you have 4 GPU server with 4 GPUs per server, then you should set it to [
    4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16
    ]
Change to fit your resources. Note that the changing period shall be a multiple of the number of GPUs per machine.

1. In local's terminal, run `./AE/3_Sharing_inference_and_training/run_test3.sh`. 
2. change three controller.json files' inference_workload_s to [0, 0] run `./AE/3_Sharing_inference_and_training/run_test3_bound.sh`
3. change settings/controller.josn file's inference_workload_s to [max_gpu_count] run `./AE/3_Sharing_inference_and_training/run_test3_gpu_bound.sh`
4. When all the script finished(without error), run `python ./AE/3_Sharing_inference_and_training/gather_result.py` to get throughput result and run `python ./AE/3_Sharing_inference_and_training/drawplot.py` to plot. The plot will be saved to  
./AE/3_Sharing_inference_and_training/{system_type}_utilization.png

### Reducing memory usage

This is only a stimulation test.
In local's terminal, run `python ./AE/4_Reducing_memory_usage/drawplot.py`. The plot will be saved to  
./AE/4_Reducing_memory_usage/fig10.png
