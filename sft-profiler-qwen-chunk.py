#####################################################################
# torchrun --standalone --nproc_per_node=2 smollm2-instruct-limo.py #
#####################################################################

from dataclasses import dataclass
import inspect
import math
import os
import time
import numpy as np
import csv

import torch
from torch.autograd.profiler import record_function

from transformers import AutoTokenizer

from datetime import datetime, timedelta
import logging
import socket

from limo import LIMODataLoader
from qwen_chunk import Qwen
# from any_model import myLLM

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
exp_name = 'chunk'

tokenizer = AutoTokenizer.from_pretrained(model_name) 

# autodetect GPU
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  device = "mps"
print(f"using {device}")

# set random seeds for reproducibility
torch.manual_seed(8888)
if torch.cuda.is_available():
  torch.cuda.manual_seed(8888)
  torch.set_float32_matmul_precision("high")

@dataclass
class QwenConfig:
  """
  defaults from Qwen/Qwen2.5-1.5B config
  """
  embed_dim : int = 1536        # hidden_size
  nbr_layer : int = 28          # num_hidden_layers
  nbr_heads : int = 12          # num_attention_heads 
  nbr_kv_heads : int = 2        # num_key_value_heads
  mlp_hidden_dim : int = 8960   # intermediate_size
  rms_norm_eps: float = 1e-6
  rope_theta : int = 1000000
  block_size: int = 32768       # max_position_embeddings
  vocab_size: int = 151936

@dataclass
class OptimConfig:
  device: str 
  max_itr: int
  batch_size: int = 2
  grad_accum_steps: int = 2
  grad_checkpointing: bool = True
  max_lr: float = 1e-5
  weight_decay: float = 1e-2
  
  def __post_init__(self):
    self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
    self.max_decay_itr = self.max_itr
    self.min_lr = self.max_lr / 10.
    self.use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and self.device_type == 'cuda'
    self.warmup_itr = self.max_itr // 100

  def cosine_decay(self, itr):
    if itr < self.warmup_itr:
      lr = self.max_lr * (itr+1) / self.warmup_itr
    elif itr > self.max_decay_itr:
      lr = self.min_lr
    else:  # cosine decay for remaining 90%
      decay_frac = (itr - self.warmup_itr) / (self.max_itr - self.warmup_itr)
      lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (math.cos(math.pi * decay_frac) + 1.) 
    return lr

opt_config = OptimConfig(device=device, max_itr=10)

# set up profiling
logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def save_memory_usage_to_csv(prof, filename, device_type="cuda"):
    events = prof.key_averages(group_by_input_shape=True)

    rows = []
    for evt in events:
        row = {
            "Name": evt.key,
            # "Input Shapes": str(evt.input_shapes),
            "CPU Mem (bytes)": getattr(evt, "cpu_memory_usage", 0),
            "Self CPU Mem (bytes)": getattr(evt, "self_cpu_memory_usage", 0),
            "# of Calls": evt.count,
            "CPU Time Total (us)": evt.cpu_time_total,
            "Self CPU Time (us)": evt.self_cpu_time_total,
        }

        # if device_type == "cuda":
        #     row["CUDA Mem (bytes)"] = getattr(evt, "cuda_memory_usage", 0)
        #     row["Self CUDA Mem (bytes)"] = getattr(evt, "self_cuda_memory_usage", 0)
        #     row["CUDA Time Total (us)"] = getattr(evt, "cuda_time_total", 0)
        #     row["Self CUDA Time (us)"] = getattr(evt, "self_cuda_time_total", 0)

        rows.append(row)

    # Sort by CUDA or CPU memory usage
    sort_key = "CPU Mem (bytes)"

    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    fieldnames = rows[0].keys()
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows) 


def trace_handler(prof: torch.profiler.profile):
  # Prefix for file names.
  # host_name = socket.gethostname()
  # host_name = 'chunk'
  timestamp = datetime.now().strftime(TIME_FORMAT_STR)
  file_prefix = f"{exp_name}_{timestamp}_qwen"

  # Construct the trace file.
  prof.export_chrome_trace(f"{file_prefix}.json.gz")

  # Construct the memory timeline file.
  prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
#     on_trace_ready=trace_handler,
# ) as prof:
#   prof.step()
# with record_function("## initialization ##"):
# instantiate model
model = Qwen(QwenConfig())
model.bfloat16()
if opt_config.grad_checkpointing:
  model.grad_enable_checkpointing()
# model = myLLM(model_name)
model.to(device)
# model = torch.compile(model)

# instantiate optimizer
extra_args = dict(fused=True) if opt_config.use_fused else dict()
optimizer = torch.optim.AdamW(model.parameters(), **extra_args)

# data loaders
train_loader = LIMODataLoader(tokenizer, opt_config.batch_size)

# set up logging
log_dir = "yk_logs"
os.makedirs(log_dir, exist_ok=True)
train_log = os.path.join(log_dir, f"{exp_name}_train_log.txt")
with open(train_log, "w") as f:
  f.write(f" iter | train loss | walltime (sec) | ktoks/sec | grad norm\n")

# training loop
checkpoint_int, print_int = 1000, 1
eval_int = print_int

# torch.cuda.memory._record_memory_history(max_entries=100000)

# file_prefix = 'test-chunk'

with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=4, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
       on_trace_ready=trace_handler,
   ) as prof:
    
    for itr in range(opt_config.max_itr):
        prof.step()
        # train
        model.train()
        min_t0 = time.time()
        
        # gradient accumulation loop
        cum_loss, batch_tokens = 0., 0
        for min_itr in range(opt_config.grad_accum_steps):
            # prof.step()
            # print("Allocated:", torch.cuda.memory_allocated(device))
            # print("Reserved:", torch.cuda.memory_reserved(device))
            # with record_function("## forward YYY ##"):
            x, y = train_loader.get_batch()
            x, y = x.to(device), y.to(device)
            # print(x.size(), y.size())
            # from IPython import embed
            # embed()
            # if torch.cuda.is_available():
            with torch.autocast(device_type=opt_config.device_type, dtype=torch.bfloat16):
                if exp_name == 'fb_chunk':
                    loss = model.forward_backward(x, y, reduction='sum')
                else:
                    loss = model.forward(x, y, reduction='sum', chunk= (exp_name == 'chunk'))
            # else:
                # loss = model(x, y, reduction='sum')
            if exp_name == 'fb_chunk':
                cum_loss += loss
            else:
                cum_loss += loss.detach()
                
            # print('\n\n' + 10*'==' + '\n\n')
            # print("Allocated:", torch.cuda.memory_allocated(device))
            # print("Reserved:", torch.cuda.memory_reserved(device))
            # with record_function("## backward ##"):
            if exp_name != 'fb_chunk':
                loss.backward()
            batch_tokens += (y >= 0).sum()
    
        # with record_function("## update ##"):
        # normalize loss and gradient by the number of (label) tokens in the batch
        avg_loss = cum_loss / batch_tokens
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= batch_tokens
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update parameters
        lr = opt_config.cosine_decay(itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # from IPython import embed
        # embed()
        # grad = model.transformer.layers[15].self_attn.q_proj.weight.grad
        # if grad is not None:
        #     print("Gradient for q_proj.weight:")
        #     print("Gradient norm:", grad.norm().item())
        # else:
        #     print("No gradient computed for q_proj.weight")

        # grad_check = model.transformer.layers[15].mlp.gate_proj.weight.grad.to(torch.float32).detach().cpu().numpy()[:100]
        # grad_norm = (grad_check**2).sum()
        # np.save(f'{exp_name}_{itr}', grad_check)
        grad_norm = 0.
        optimizer.step()
        optimizer.zero_grad()
    
        # logging
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if (itr % print_int == 0):
            min_t1 = time.time()
            min_dt = min_t1 - min_t0
            TPS = batch_tokens / min_dt 
            print(f"iter {itr} | train loss: {avg_loss.item():.4f} | walltime: {min_dt:.2f} sec | ktoks/sec: {TPS/1e3:.2f} | grad norm: {grad_norm:.2f}")
            with open(train_log, "a") as f:
                f.write(f"{itr:5d} |    {avg_loss.item():.4f} |           {min_dt:.2f} |      {TPS/1e3:.2f} |      {grad_norm:.2f}\n")

# save_memory_usage_to_csv(prof, f"{exp_name}_memory_profile.csv", device_type=device)

# prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
# torch.cuda.memory._dump_snapshot("test-profile.pkl")
# torch.cuda.memory._record_memory_history(enabled=None)

    # # save checkpoint
    # if (itr % checkpoint_int == 0) or (itr == opt_config.max_itr - 1):
    #   checkpoint_path = os.path.join(log_dir, f"checkpoint_{itr:04d}.pth")
    #   checkpoint = {
    #     "iter": itr,
    #     "model_config": model.config,
    #     "model_state": model.state_dict(),
    #     "optimizer_config": opt_config,
    #     "optimizer_state": optimizer.state_dict(),
    #     "train loss": avg_loss.item(),
    #   }
    #   torch.save(checkpoint, checkpoint_path)
