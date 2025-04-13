#####################################################################
# torchrun --standalone --nproc_per_node=2 smollm2-instruct-limo.py #
#####################################################################

from dataclasses import dataclass
import inspect
import math
import os
import time

import torch
from torch.autograd.profiler import record_function

from transformers import AutoTokenizer

from datetime import datetime, timedelta
import logging
import socket

from limo import LIMODataLoader
from qwen import Qwen
# from any_model import myLLM

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'

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
  grad_checkpointing_frequency: int = 1
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

def trace_handler(prof: torch.profiler.profile):
  # Prefix for file names.
  host_name = socket.gethostname()
  timestamp = datetime.now().strftime(TIME_FORMAT_STR)
  file_prefix = f"{host_name}_{timestamp}_og-qwen"

  # Construct the trace file.
  prof.export_chrome_trace(f"{file_prefix}.json.gz")

  # Construct the memory timeline file.
  prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=trace_handler,
) as prof:
  prof.step()
  with record_function("## initialization ##"):
    # instantiate model
    model = Qwen(QwenConfig())
    model.bfloat16()
    if opt_config.grad_checkpointing:
      model.grad_enable_checkpointing(opt_config.grad_checkpointing_frequency)
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
    train_log = os.path.join(log_dir, f"train_log.txt")
    with open(train_log, "w") as f:
      f.write(f" iter | train loss | walltime (sec) | ktoks/sec\n")

  # training loop
  checkpoint_int, print_int = 1000, 1
  eval_int = print_int
  for itr in range(opt_config.max_itr):
    # train
    model.train()
    min_t0 = time.time()

    # gradient accumulation loop
    cum_loss, batch_tokens = 0., 0
    for min_itr in range(opt_config.grad_accum_steps):
      prof.step()
      with record_function("## forward ##"):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        # print(x.size(), y.size())
        # from IPython import embed
        # embed()
        if torch.cuda.is_available():
          with torch.autocast(device_type=opt_config.device_type, dtype=torch.bfloat16): 
            _, loss = model(x, y, reduction='sum')
        else:
          _, loss = model(x, y, reduction='sum')
        cum_loss += loss.detach()
      with record_function("## backward ##"):
        loss.backward()
        batch_tokens += (y >= 0).sum()

    with record_function("## update ##"):
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
      optimizer.step()
      optimizer.zero_grad()

    # logging
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    if (itr % print_int == 0):
      min_t1 = time.time()
      min_dt = min_t1 - min_t0
      TPS = batch_tokens / min_dt 
      print(f"iter {itr} | train loss: {avg_loss.item():.4f} | walltime: {min_dt:.2f} sec | ktoks/sec: {TPS/1e3:.2f}")
      with open(train_log, "a") as f:
        f.write(f"{itr:5d} |    {avg_loss.item():.4f} |           {min_dt:.2f} |      {TPS/1e3:.2f}\n")
    
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
