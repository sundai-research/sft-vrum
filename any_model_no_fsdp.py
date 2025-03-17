from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

class myLLM(torch.nn.Module):
    def __init__(self, model_name, gradient_checkpointing=True):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_cache = False, torch_dtype=torch.bfloat16)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Important settings for memory efficiency
        # self.chunk_size = 2048  # Adjust based on GPU memory
        # self.gradient_checkpointing = gradient_checkpointing

    def forward_chunk(self, ids, targets=None, attention_mask=None, reduction='mean'):
        B, T = ids.size()

        if attention_mask is None:
            attention_mask = torch.ones_like(ids)
            
        # Get last hidden states from the model's native forward
        # This leverages the model's internal optimizations
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        last_hidden_state = self.model.model(
            input_ids=ids,
            attention_mask=attention_mask,
            output_hidden_states=False,  # We need the hidden states
            use_cache=False,            # Disable KV cache for memory efficiency
            return_dict=True
        ).last_hidden_state

        # from IPython import embed
        # embed()
        
        # Get the last hidden state from model outputs
        # last_hidden_state = outputs.hidden_states[-1]

        # print(last_hidden_state.size())
        # Free memory from other hidden states if not needed
        # del outputs.hidden_states
            
        # Process in chunks for logits and loss calculation
        chunk_size = min(2048, T)
        
        if targets is not None:
            total_loss = 0
            last_chunk_logits = None
            
            # Process lm_head and loss in chunks
            for i in range(0, T, chunk_size):
                end_idx = min(i+chunk_size, T)
                
                # Get chunk of hidden states and compute logits
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                chunk_hidden = last_hidden_state[:, i:end_idx, :]
                chunk_logits = self.model.lm_head(chunk_hidden)
                
                # Save last chunk logits for return
                if i + chunk_size >= T:
                    last_chunk_logits = chunk_logits
                
                # Calculate loss for this chunk
                if targets is not None:
                    chunk_targets = targets[:, i:end_idx].contiguous()
                    chunk_loss = F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.size(-1)),
                        chunk_targets.reshape(-1),
                        reduction='sum'
                    )
                    total_loss += chunk_loss
                
                # Free memory
                del chunk_logits
                del chunk_hidden
                torch.cuda.empty_cache()  # Optional: explicit memory cleanup
            
            # Calculate final loss based on reduction type
            if reduction == 'mean':
                loss = total_loss / (B * T)
            else:
                loss = total_loss
                
            return last_chunk_logits, loss
            
        else:
            # For inference, also process lm_head in chunks
            # all_logits = []
            
            # for i in range(0, T, chunk_size):
            #     end_idx = min(i+chunk_size, T)
                
            #     # Process this chunk
            #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            #         chunk_hidden = last_hidden_state[:, i:end_idx, :]
            #         chunk_logits = self.model.lm_head(chunk_hidden)
                
            #     all_logits.append(chunk_logits)
                
            #     # Free memory
            #     del chunk_hidden
            
            # # Concatenate all logits
            # logits = torch.cat(all_logits, dim=1)

            logits = self.model.lm_head(x)
            
            return logits