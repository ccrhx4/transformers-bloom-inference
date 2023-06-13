from ..constants import DS_INFERENCE, DS_ZERO, HF_ACCELERATE, HF_CPU
from .model import Model, get_hf_model_class, load_tokenizer
import os

def get_model_class(deployment_framework: str):
    if deployment_framework == HF_ACCELERATE:
        from .hf_accelerate import HFAccelerateModel

        return HFAccelerateModel
    elif deployment_framework == HF_CPU:
        from .hf_cpu import HFCPUModel

        return HFCPUModel
    elif deployment_framework == DS_INFERENCE:
        from .ds_inference import DSInferenceModel

        return DSInferenceModel
    elif deployment_framework == DS_ZERO:
        from .ds_zero import DSZeROModel

        return DSZeROModel
    else:
        raise ValueError(f"Unknown deployment framework {deployment_framework}")


def start_inference_engine(deployment_framework: str) -> None:
    if deployment_framework in [DS_INFERENCE, DS_ZERO]:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        import deepspeed.comm as dist
        import torch

        # deepspeed.init_distributed("nccl")
        deepspeed.init_distributed(get_accelerator().communication_backend_name())
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        print(str(local_rank))
        device = torch.device('xpu', local_rank)
        print("start inference engine", str(device))
        x = torch.ones([4, 1, 14336], device=device, dtype=torch.bfloat16)
        dist.all_reduce(x)
