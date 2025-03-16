from datasets import load_dataset
from transformers import AutoTokenizer

import torch

system_prefix = "<|im_start|>system\n"
user_prefix = "<|im_start|>user\n"
assistant_prefix = "<|im_start|>assistant\n"
suffix = "<|im_end|>\n"
N_SAMPLES = 50
MAX_LEN = 16000

def load_LIMO_dataset(tokenizer):
  """
  download and tokenize the LIMO dataset
  """
  def format_LIMO_example(example):
    SYS_PROMPT = """Let's think step by step and output the final answer within \\boxed{}. """
    input_text = tokenizer.apply_chat_template([{'role': 'user', 'content': SYS_PROMPT + example['question']}], tokenize=False, add_generation_prompt=True)
    output_text = example['solution'] + tokenizer.eos_token
    tokens = tokenizer(input_text + output_text)['input_ids']
    len_seq = len(tokens)
    n_no_loss = len(tokenizer(input_text)['input_ids'])
    labels = [-100]*n_no_loss + tokens[n_no_loss:]
    return {"tokens": tokens[:MAX_LEN], "labels": labels[:MAX_LEN], "len": len_seq}
  
  LIMO = load_dataset("GAIR/LIMO", split="train")
  LIMO = LIMO.map(format_LIMO_example)
  LIMO = LIMO.sort("len", reverse=True)
  LIMO = LIMO.select(range(N_SAMPLES))
  return LIMO

class LIMODataLoader:
  """
  data loader for the LIMO dataset that supports distributed training
  """
  def __init__(self, tokenizer, batch_size, rank=1, world_size=1):
    self.batch_size = batch_size
    self.data = load_LIMO_dataset(tokenizer)
    self.rank = rank
    self.tokenizer = tokenizer
    self.world_size = world_size

    self.reset()

  def reset(self):
    self.example_idx = self.batch_size * self.rank

  def get_batch(self):
    batch = self.data.select(range(self.example_idx, self.example_idx + self.batch_size))

    # pad token and label sequences so that they are all the same length
    max_len = max(d["len"] for d in batch)
    x = torch.tensor([d["tokens"][:-1] + [self.tokenizer.pad_token_id] * (max_len - len(d["tokens"])) for d in batch])
    y = torch.tensor([d["labels"][1:] + [-100] * (max_len - len(d["labels"])) for d in batch])

    self.example_idx += self.batch_size * self.world_size
    if (self.example_idx + self.batch_size * self.world_size + 1) > len(self.data):
      self.reset()
    return x, y
