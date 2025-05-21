# 0. prepare model_list.txt and requests.txt

``` bash
PYTHONPATH=./ python source/deployment/generate_model_list.py source/deployment/storage_list.txt source/deployment/model_list_seed.txt 10 source/mps/model_list.txt source/deployment/model_distribution_list.txt

PYTHONPATH=./ python source/workload/workload.py source/mps/model_list.txt 100 1 0.9 test_requests.txt
```

# 1. launch load balancer
`PYTHONPATH=./ python source/mps/load_balancer.py`

If enable training for filling
`PYTHONPATH=./ python source/mps/load_balancer.py xxx-any`

# 2. launch server agents
# 2.1 load models to shm
`PYTHONPATH=./ python source/mps/load_models.py source/mps/model_list.txt source/mps/model_sizes.txt`

# 2.2 launch server agents
`PYTHONPATH=./ python source/mps/server_agent.py --lb-port 8778 --lb-ip 127.0.0.1 --size-list source/mps/model_sizes.txt`

# 3. send client requests
```
PYTHONPATH=./ python source/client/client_concurrent.py test_requests.txt 127.0.0.1 8777 4

PYTHONPATH=./ python source/client/client.py test_requests-1.txt 127.0.0.1 8777
```