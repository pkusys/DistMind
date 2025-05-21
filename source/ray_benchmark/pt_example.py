from ray import serve
import ray
import os
import time

from io import BytesIO
from PIL import Image
import random
from ray.serve.api import start

import torch
from torchvision import transforms
from torchvision.models import resnet152, vgg16_bn, densenet201, inception_v3
from transformers import BertModel, GPT2Model
import threading
import numpy as np

from fastapi import FastAPI, Request

app = FastAPI()
# ray.init(address="auto", _redis_password='5241590000000000', namespace="model_serve")
serve.start(detached=True, http_options={"host": "0.0.0.0"})

def get_random_data(batch_size):
    data = torch.rand([batch_size, 3, 224, 224], dtype=torch.float)
    target = torch.randint(0, 1000, [batch_size])
    return data, target

@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class ImageModel:

    
    @app.get("/train")
    def training(self):
        """"""
        with self.status_lock:
            self.training_flag = True
        return {"result": 'ok'}
    
    @app.get("/training_batch_count")
    def training_batch_count(self):
        return {"count": self.training_batch_n}

    @app.get("/inference_count")
    def inference_count(self):
        return {"count": self.inf_task_count}
    
    def training_thread_func(self):
        """"""
        bs = 64
        device = f"cuda:{self.gpu_id}"
        print(f'training would happend on device {device}')
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        while not self.shutdown:
            if not self.training_flag:
                time.sleep(0.005)
            else:
                self.training_model.train()
                # self.training_model = self.training_model.cuda()
                self.training_model = self._prepare_env(len(self.model_pool) - 1)[0]
                optimizer = torch.optim.SGD(self.training_model.parameters(), lr=0.1)
                print(f'gpu-{self.gpu_id} start training')
                while self.training_flag:
                    with self.status_lock:
                        self.training_running = True

                    data, target = get_random_data(bs)
                    data = data.cuda()
                    target = target.cuda()
                    output = self.training_model(data)
                    # print(f'gpu-{self.gpu_id} get output')
                    loss = loss_fn(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.training_batch_n += 1

                with self.status_lock:
                    self.training_running = False
                print(f"gpu-{self.gpu_id} stop training")
                del optimizer

    @app.get("/incr")
    def incr(self):
        """ for testing"""
        self.count += 1
        return {"count": self.count}

    def __del__(self, ):
        self.shutdown = True
    
    def log(self, msg):
        print(f"{self.gpu_id}: {msg}")    

    def __init__(self, n_model_variants=10, model_name="res"):
        self.gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

        self.shutdown = False
        self.training_flag = False
        self.training_running = False
        self.training_batch_n = 0
        self.status_lock = threading.RLock()
        self.count = 0  # Initialize count for the incr method

        self.model_on_gpu = -1 # no model on GPU  
        self.model_name = model_name
        self.model_loading = True  # 加载中标志位

        self.training_model = resnet152()
        self.training_thd = threading.Thread(target=self.training_thread_func, args=())
        self.training_thd.start()
        
        # 后台线程加载模型池
        self.model_load_thd = threading.Thread(
            target=self._load_model_pool, args=(n_model_variants, model_name)
        )
        self.model_load_thd.start()

        # 图像预处理器
        self.preprocessor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t[:3, ...]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.inf_task_count = 0
        # [model, model-name, memory-status]
        self.model_pool = []
        self.model_pool.append([self.training_model, 'resnet152-train', 'CPU'])

    # 添加查询模型状态的 API
    @app.get("/model_ready")
    def model_ready(self):
        return {"ready": not self.model_loading}

    def _load_model_pool(self, n_model_variants, model_name):
        model_choices = []
        if model_name == "res":
            model_choices.append((resnet152, "resnet152-inf"))
        elif model_name == "inc":
            model_choices.append((inception_v3, "inception_v3-inf"))
        elif model_name == "den":
            model_choices.append((densenet201, "densenet201-inf"))
        elif model_name == "bert":
            model_choices.append((BertModel, "bert-base-inf"))
        elif model_name == "gpt":
            model_choices.append((GPT2Model, "gpt2-inf"))

        for i in range(n_model_variants):
            model_func, name = random.choice(model_choices)
            if name == "bert-base-inf":
                m = BertModel.from_pretrained('bert-base-uncased').eval()
            elif name == "gpt2-inf":
                m = GPT2Model.from_pretrained('gpt2').eval()
            else:
                m = model_func(pretrained=True).eval()
            self.model_pool.append([m, f"{name}-{i}", 'CPU'])

        self.model_loading = False
        print("Model pool loading complete.")

    
    def _wait_training_end(self):
        with self.status_lock:
                training_running = self.training_running
        while training_running:
            time.sleep(0.001)
            with self.status_lock:
                training_running = self.training_running
    
    def _unload_model_on_gpu(self, model_idx):
        """"""
        m, name, status = self.model_pool[model_idx]
        self.log(f"before unload {model_idx}:{name}, device {next(m.parameters()).device}")
        assert status == 'GPU'
        m = m.to('cpu')
        self.log(f"after unload {model_idx}:{name}, device {next(m.parameters()).device}")
        self.model_pool[model_idx] = [m, name, 'CPU']
        self.model_on_gpu = -1

    def _load_model_to_gpu(self, model_idx):
        m, name, status = self.model_pool[model_idx]
        assert status == "CPU"
        m = m.cuda()
        self.model_pool[model_idx] = [m, name, 'GPU']
        self.model_on_gpu = model_idx

    def _prepare_env(self, model_idx):
        if self.model_on_gpu == model_idx:
            return self.model_pool[model_idx]

        elif self.model_on_gpu == len(self.model_pool) - 1:
            # the model_idx != self.model_on_gpu
            # in training 
            with self.status_lock:
                self.training_flag = False
            self._wait_training_end()

        if self.model_on_gpu != -1:
            self._unload_model_on_gpu(self.model_on_gpu)

        self._load_model_to_gpu(model_idx)

        return self.model_pool[model_idx]

    @app.get("/inference/{model_idx}")
    def inference(self, model_idx):
        """"""
        start_t = time.time() * 1e3
        model_idx = int(model_idx)
        inf_bs = 8

        input_shapes = {
            'res': [-1, 3, 224, 224],
            'inc': [-1, 3, 299, 299],
            'den': [-1, 3, 224, 224],
            'bert': [-1, 512],
            'gpt': [-1, 512]
        }
        shape = input_shapes[self.model_name]
        shape[0] = inf_bs
        if self.model_name == "bert":
            fake_data = torch.randint(0, 30522, shape, dtype=torch.long)
        elif self.model_name == "gpt":
            fake_data = torch.randint(0, 50257, shape, dtype=torch.long)
        else:
            fake_data = torch.rand(shape, dtype=torch.float32)

        m, name, status = self._prepare_env(model_idx)
        assert status == "GPU", "model inference must on GPU"
        fake_data = fake_data.cuda()    
        with torch.no_grad():
            outputs = m(fake_data)
        end_t = time.time() * 1e3
        self.log(f"{name} inference done, cost {end_t - start_t} ms")

        self.inf_task_count += 1

        if self.model_name == "bert":
            vec = outputs.pooler_output  # [B, hidden_dim]
            class_index = torch.argmax(vec, dim=1).tolist()
        elif self.model_name == "gpt":
            vec = outputs.last_hidden_state[:, -1, :]  # [B, hidden_dim]
            class_index = torch.argmax(vec, dim=1).tolist()
        else:
            class_index = torch.argmax(outputs, dim=1).tolist()
        return {"class_index": class_index}

    # @app.post("/")
    # async def __call__(self, starlette_request):
    # @app.post("/query")
    # async def serve(self, req: Request):
    #     start_t = time.time()
    #     image_payload_bytes = await req.body()
    #     # pil_image = Image.open(BytesIO(image_payload_bytes))
    #     image_data = torch.from_numpy(
    #         np.frombuffer(image_payload_bytes, dtype=np.float32)).cuda()
    #     image_data = image_data.reshape(-1, 3, 224,224)
    #     print("[1/3] Parsed image data: {}".format(image_data.size()))

    #     # pil_images = [pil_image]  # Our current batch size is one
    #     # input_tensor = torch.cat(
    #     #     [self.preprocessor(i).unsqueeze(0) for i in pil_images]).cuda()
    #     input_tensor = image_data
    #     print("[2/3] Images transformed, tensor shape {}".format(
    #         input_tensor.shape))
    #     with torch.no_grad():
    #         _id = self.cur_model_idx
    #         output_tensor = self.model_pool[_id](input_tensor)

    #     print(f"[3/3] {self.model_types[self.cur_model_idx]} Inference done! time {(time.time() - start_t)*1e3}ms")
    #     return {"class_index": torch.argmax(output_tensor, dim=1).tolist()}


# ImageModel.options(name="model1", route_prefix="/image_predict1").deploy()
# ImageModel.options(name="model2", route_prefix="/image_predict2").deploy()
import argparse
argparse = argparse.ArgumentParser()
argparse.add_argument("--n_gpus", type=int, default=2, help="number of gpus")
argparse.add_argument("--n_model_variants", type=int, default=10, help="number of model variants")
argparse.add_argument("--model_name", type=str, default="res", help="model name")
args = argparse.parse_args()
n_gpus = args.n_gpus
n_model_variants = args.n_model_variants
model_name = args.model_name
for i in range(n_gpus):
    # First create a deployment object with options
    deployment = ImageModel.options(
        name=f"GPU-{i}", 
        route_prefix=f"/gpu-{i}"
    )
    # Then deploy it with the required constructor parameters
    deployment.deploy(n_model_variants=n_model_variants, model_name=model_name)

while True:
    time.sleep(10)
    print(serve.list_deployments())