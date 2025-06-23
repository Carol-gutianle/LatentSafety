import random
import torch
import seaborn as sns
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from data.dataset import apply_alpaca_template
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed):
    random.seed(seed)                 
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)

def read_data_from_json(json_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_data_to_json(data, save_path):
    import json
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_data_from_jsonlines(jsonl_path):
    import jsonlines
    data = []
    with jsonlines.open(jsonl_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data

def get_model_and_tokenizer(model_name_or_path, mode='pca', device_map="auto"):
    dtype = torch.float32 if mode == 'pca' else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    if 'vicuna' or 'lama' in model_name_or_path:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def prepare_data(pos_data_path, neg_data_path, num_samples):
    pos_data = load_dataset('parquet', data_files=[pos_data_path])['train']
    neg_data = load_dataset('csv', data_files=[neg_data_path])['train']
    pos_data = [apply_alpaca_template(row['instruction'], row['input']) for row in pos_data]
    neg_data = [row['target'] for row in neg_data]
    pos_data = random.sample(pos_data, num_samples)
    neg_data = random.sample(neg_data, num_samples)
    return pos_data, neg_data

    
def angle_between(a: torch.Tensor, b: torch.Tensor):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    
    cos_theta = (a_norm * b_norm).sum(dim=-1).clamp(-1.0, 1.0)
    theta_rad = torch.acos(cos_theta)
    theta_deg = theta_rad * (180.0 / math.pi) 
    return theta_deg
    
def kl_divergence(p, q):
    import torch
    p = torch.tensor(p)
    q = torch.tensor(q)
    return torch.sum(p * torch.log(p / q))

def cos_similarity(p, q):
    p = p.clone().detach() if isinstance(p, torch.Tensor) else torch.tensor(p)
    q = q.clone().detach() if isinstance(q, torch.Tensor) else torch.tensor(q)
    p = p.squeeze()
    q = q.squeeze()
    if len(p.shape) == 2:
        p = p[0, :]
    if len(q.shape) == 2:
        q = q[0, :]
    return torch.dot(p, q) / (torch.norm(p) * torch.norm(q) + 1e-8)

def plot_attention_heatmap(attn_matrix, tokens, layer_idx, head_idx):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_matrix, annot=False, fmt=".2f", cmap='viridis', cbar=True, ax=ax)
    plt.title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300)
    plt.show()
    
'''
Dataloader related tools
'''
def fetch_pos_neg_data(num_samples=50):
    set_seed(42)
    pos_data = load_dataset('parquet', data_files=['../data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet'])['train']
    neg_data = load_dataset('csv', data_files=['../data/advbench/harmful_strings.csv'])['train']
    pos_data = [apply_alpaca_template(row['instruction'], row['input']) for row in pos_data]
    if 'row' in neg_data[0]:
        neg_data = [row['target'] for row in neg_data]
    else:
        neg_data = [row['prompt'] for row in neg_data]
    pos_samples = random.sample(pos_data, num_samples)
    neg_samples = random.sample(neg_data, num_samples)
    return pos_samples, neg_samples