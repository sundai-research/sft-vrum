from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)

class RoPE(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.theta = config.rope_theta
    self.head_dim = config.embed_dim // config.nbr_heads
    inv_freqs = self.theta ** -(torch.arange(0, self.head_dim, 2).float() / self.head_dim)
    self.register_buffer("inv_freqs", inv_freqs)
  
  @torch.no_grad()
  def forward(self, q, k, pos_ids=None):
    B, _, T, _ = q.size()
    if pos_ids is None:
      pos_ids = torch.arange(T, device=q.device).view(T,1).expand(B,T,1)
    Theta = torch.einsum("btd,d->btd", pos_ids.view(B,T,1), self.inv_freqs)
    tmp = torch.cat((Theta, Theta), dim=-1)
    cos = tmp.cos().view(B,1,T,-1)
    sin = tmp.sin().view(B,1,T,-1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

class MultiheadAttention(nn.Module):
  """
  PyTorch's MultiheadAttention does not support grouped query attention (GQA) and rotary position embeddings (RoPE), so we have to roll our own.
  """
  def __init__(self, config, rope):
    super().__init__()
    self.embed_dim = config.embed_dim
    self.nbr_heads = config.nbr_heads
    self.nbr_kv_heads = config.nbr_kv_heads
    assert self.embed_dim % self.nbr_heads == 0, "embed_dim must be divisible by nbr_heads"
    assert self.nbr_heads % self.nbr_kv_heads == 0, "nbr_heads must be divisible by nbr_kv_heads"
    self.head_dim = self.embed_dim // self.nbr_heads 
    assert self.head_dim % 2 == 0, "head_dim must be even (for RoPE)"
    self.q_proj = nn.Linear(self.embed_dim, self.head_dim * self.nbr_heads, bias=True)
    self.k_proj = nn.Linear(self.embed_dim, self.head_dim * self.nbr_kv_heads, bias=True)
    self.v_proj = nn.Linear(self.embed_dim, self.head_dim * self.nbr_kv_heads, bias=True)
    self.o_proj = nn.Linear(self.head_dim * self.nbr_heads, self.embed_dim, bias=False)
    self.rope = rope
  
  def forward(self, x):
    B, T, C = x.size()
    q = self.q_proj(x).view(B, T, self.nbr_heads, self.head_dim).transpose(1,2) # (B, nbr_heads, T, head_dim=C/nbr_heads) 
    k = self.k_proj(x).view(B, T, self.nbr_kv_heads, self.head_dim).transpose(1,2) # (B, nbr_kv_heads, T, head_dim=C/nbr_heads) 
    v = self.v_proj(x).view(B, T, self.nbr_kv_heads, self.head_dim).transpose(1,2)
    q, k = self.rope(q, k)
    x = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    x = x.transpose(1,2).contiguous().view(B,T,C) # stack head outputs
    x = self.o_proj(x)
    return x

class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.gate_proj = nn.Linear(config.embed_dim, config.mlp_hidden_dim, bias=False)
    self.up_proj = nn.Linear(config.embed_dim, config.mlp_hidden_dim, bias=False)
    self.down_proj = nn.Linear(config.mlp_hidden_dim, config.embed_dim,bias=False)
    self.act_fn = nn.SiLU()

  def forward(self, x):
    x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return x
  
class Block(nn.Module):

  def __init__(self, config, rope):
    super().__init__()
    self.input_layernorm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
    self.self_attn = MultiheadAttention(config, rope)
    self.post_attention_layernorm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
    self.mlp = MLP(config)
  
  def forward(self, x):
    x = x + self.self_attn(self.input_layernorm(x))
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x

class Qwen(nn.Module): 

  def __init__(self, config):
    super().__init__()
    self.config = config # save config input so that it is convenient to inspect it later
    self.rope = RoPE(config)
    self.transformer = nn.ModuleDict(dict(
      embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim),
      layers = nn.ModuleList([Block(config, self.rope) for _ in range(config.nbr_layer)]),
      norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps),
    ))
    self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    self.transformer.embed_tokens.weight = self.lm_head.weight # share token embedding and lm_head weights
    self.grad_checkpointing = False

  def grad_enable_checkpointing(self):
    self.grad_checkpointing = True

  @classmethod
  def load_checkpoint(cls, model, config):
    model = cls(config)
    sd, hf_sd = model.state_dict(), hf_model.state_dict()
    for k in sd.keys():
      assert sd[k].shape == hf_sd[hf_k].shape, f"shape mismatch for {k}: {sd[k].shape} vs {hf_sd[hf_k].shape}"
      with torch.no_grad():
        sd[k].copy_(hf_sd[hf_k])
    return model

  @classmethod
  def load_hf_model(cls, config, hf_model):
    model = cls(config)
    sd, hf_sd = model.state_dict(), hf_model.state_dict()
    for k in sd.keys():
      if "transformer" in k:
        hf_k = k.replace("transformer", "model")
      else:
        hf_k = k
      skip_keys = ["rope", "triu_mask"]
      if any(s in k for s in skip_keys):
        continue  # skip RoPE values
      assert sd[k].shape == hf_sd[hf_k].shape, f"shape mismatch for {k}: {sd[k].shape} vs {hf_sd[hf_k].shape}"
      with torch.no_grad():
        sd[k].copy_(hf_sd[hf_k])
    return model
  
  def forward(self, ids, targets=None, reduction='mean'):
    B, T = ids.size()
    assert T <= self.config.block_size, f"input sequence length {T} exceeds block size {self.config.block_size}"
    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    x = self.transformer.embed_tokens(ids)
    if self.grad_checkpointing and self.training:
      for block in self.transformer.layers:
        x = checkpoint(block, x, use_reentrant=False)
    else:
      for block in self.transformer.layers:
        x = block(x)
    x = self.transformer.norm(x)

    chunk_size = min(2048, T)
    # Calculate logits and loss with memory-efficient operations
    if targets is not None:
        # Process in chunks for logits and loss calculation
        total_loss = 0
        for i in range(0, T, chunk_size):
            end_idx = min(i+chunk_size, T)
            chunk_x = x[:, i:end_idx, :]
            chunk_logits = self.lm_head(chunk_x)
            
            if targets is not None:
                chunk_targets = targets[:, i:end_idx].contiguous()
                chunk_loss = F.cross_entropy(
                    chunk_logits.reshape(-1, chunk_logits.size(-1)),
                    chunk_targets.reshape(-1),
                    reduction='sum'
                )
                total_loss += chunk_loss
                
        if targets is not None:
            if reduction == 'mean':
                loss = total_loss / (B * T)
            else:
                loss = total_loss
            return chunk_logits, loss  # Return last chunk logits when loss is requested
    
    # For inference, process full logits
    logits = self.lm_head(x)
    return logits

      
    # logits = self.lm_head(x)
    # if targets is None:
    #   return logits
    # else:
    #   loss = F.cross_entropy(logits.view(B*T,-1), targets.view(B*T), reduction=reduction)
    #   return logits, loss

  def generate(self, ids, max_new_tokens=100, temp=1., topk=50):
    B, T = ids.size()
    with torch.no_grad():
      while ids.size(1) < max_new_tokens:
        logits = self(ids)[:,-1,:] / temp
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, topk, dim=-1)
        new_ids = torch.multinomial(topk_probs, 1)
        new_ids = torch.gather(topk_ids, -1, new_ids)
        ids = torch.cat((ids, new_ids), dim=1)
    return ids