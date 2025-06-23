'''
Scripts used for Layer-wise Adversarial Patch Training (LAPT)
'''
import os
import argparse
import pandas as pd
import torch
from trl import SFTTrainer, SFTConfig
from rich import print
from sv.intervent import fetch_layers_by_ids
from datasets import Dataset
from exp.asa import ASAEval
from sv.utils import get_model_and_tokenizer, set_seed
import traceback

def trace_var(var):
    print(f"Tracing value: {var}")
    traceback.print_stack()

class LayerConstrainedSFTTrainer(SFTTrainer):
    
    def __init__(self, seed, mode, *args, **kwargs):
        self.seed = seed
        self.mode = mode
        super().__init__(*args, **kwargs)
        
    def normalize_delta_to_preserve_stats(self, reference, delta):
        """
        Normalize each row in delta to have the same mean and std as the corresponding row in reference.

        Args:
            reference: Tensor of shape (num_samples, hidden_dim)
            delta: Tensor of shape (num_samples, hidden_dim)

        Returns:
            norm_delta: Tensor of shape (num_samples, hidden_dim)
        """
        mean_r = reference.mean(dim=1, keepdim=True)  # (num_samples, 1)
        std_r = reference.std(dim=1, keepdim=True)    # (num_samples, 1)

        mean_d = delta.mean(dim=1, keepdim=True)
        std_d = delta.std(dim=1, keepdim=True)

        norm_delta = (delta - mean_d) / (std_d + 1e-6) * std_r + mean_r
        return norm_delta

        
    def get_static_steer_vector(self, normal_hidden_states, alpha=1):
        '''
        Create random static steer_vector
        '''
        set_seed(self.seed)
        steer_vector = torch.randn_like(normal_hidden_states)
        # normal_hidden_states: (batch_size, seq_len, hidden_dim)
        for i in range(normal_hidden_states.shape[1]):
            normal_hidden_state = normal_hidden_states[:, i, :]
            random_center = torch.randn_like(normal_hidden_state)
            random_center = random_center * alpha
            random_center = random_center.to(normal_hidden_state.device)
            random_center = self.normalize_delta_to_preserve_stats(normal_hidden_state, random_center)
            steer_vector[:, i, :] = random_center
        return steer_vector
    
    def _register_training_steer_hooks(self, model, layer_idx):
        self._steer_hooks = []
        target_layers = fetch_layers_by_ids(model, layer_idx)
        for i, layer_module in enumerate(target_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    if module.training and hasattr(self, 'steer_vector_dict') and idx in self.steer_vector_dict:
                        steer_vector = self.steer_vector_dict[idx]['vector'].detach()
                        if isinstance(output, tuple):
                            out = output[0]
                            prompt_len = steer_vector.shape[1]
                            out[:, -prompt_len: , :] += steer_vector.to(out.device)
                            return (out,) + output[1:]
                        else:
                            output[:, -prompt_len, :] += steer_vector.to(output.device)
                            return output
                    return output
                return hook_fn
            hook = layer_module.register_forward_hook(make_hook(layer_idx))
            self._steer_hooks.append(hook)
    
    def _remove_steer_hooks(self):
        for hook in self._steer_hooks:
            hook.remove()
        self._steer_hooks = []
    
    def generate_steer_vector_dict(self, hidden_states, prompt_lens, layer_idx):
        '''
        vector: (batch_size, len_labels)
        '''
        steer_vector_dict = {}
        layer_hidden = hidden_states[layer_idx]
        # layer_hidden: (batch_size, seq_len, hidden_dim)
        steer_vector = layer_hidden[:, prompt_lens:, :]
        # normalize steer vector
        steer_vector = self.get_static_steer_vector(steer_vector, alpha=1.0)
        steer_vector_dict[layer_idx] = {}
        steer_vector_dict[layer_idx]['vector'] = steer_vector
        return steer_vector_dict
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        layer_idx = inputs['layer_idx']
        prompt_lens = inputs['prompt_lens']
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            self.steer_vector_dict = self.generate_steer_vector_dict(hidden_states, prompt_lens, layer_idx)
        self._register_training_steer_hooks(model, layer_idx)
        # prepare for labels
        labels = inputs['input_ids'].clone()
        inputs['labels'] = labels
        loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        self._remove_steer_hooks()
        return loss

def prepare_dataset(model_name, model_name_or_path, tokenizer):
    asaeval = ASAEval(model_name_or_path, model_name, load_dataset_only=True)
    train_dataset = asaeval.load_asa_bench(model_name, mode='train')
    conversations = []
    prompt_lens = []
    for sample in train_dataset:
        conversations.append({
            'text': sample['prompt'] + sample['original_response'],
            'layer_idx': sample['layer_idx']
        })
        prompt_lens.append(tokenizer.encode(sample['prompt'], add_special_tokens=False, return_tensors='pt').shape[1])
    return Dataset.from_dict({'text': [d['text'] for d in conversations], 'layer_idx': [d['layer_idx'] for d in conversations], 'prompt_lens': prompt_lens})
    

def main(args):
    
    model, tokenizer = get_model_and_tokenizer(args.model_name_or_path)
    data = prepare_dataset(args.model_name, args.model_name_or_path, tokenizer)

    args.max_steps = args.max_steps if args.max_steps > 0 else len(data) // args.train_size_per_gpu * args.num_train_epochs // args.gradient_accumulation_steps
    print(len(data), 'samples in total')
    
    def tokenize_function(examples):
        
        texts = examples["text"]
        
        # Tokenize full texts
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
                
        # pass layer_idx
        tokenized['labels'] = tokenized["input_ids"].clone()
        tokenized["layer_idx"] = torch.tensor(examples["layer_idx"])        
        return tokenized

    
    tokenized_dataset = data.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
            
    training_args = SFTConfig(
        per_device_train_batch_size = args.train_size_per_gpu,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = 0.1,
        lr_scheduler_type = 'cosine',
        num_train_epochs = args.num_train_epochs,
        learning_rate = args.learning_rate,
        max_steps = args.max_steps,
        bf16 = True,
        logging_steps = 1,
        optim = 'paged_adamw_32bit',
        save_strategy = "steps",
        weight_decay = 2e-6,
        eval_steps = None,
        save_steps = 100,
        output_dir = args.save_path,
        save_total_limit = 1,
        group_by_length = False,
        remove_unused_columns = False
    )
    
    def collate_with_layer_idx(batch):
        batch = {k: torch.tensor([d[k] for d in batch]) for k in batch[0].keys() if k != 'text'}
        return batch
    
    trainer = LayerConstrainedSFTTrainer(
        model = model,
        seed = 42,
        args = training_args,
        train_dataset = data,
        mode = args.mode,
        data_collator = collate_with_layer_idx
    )
    
    model.config.use_cache = False

    trainer.train()
    trainer.save_model(args.save_path)
    model.config.to_json_file(os.path.join(args.save_path, "config.json"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Llama-3.2-3B-Instruct')
    parser.add_argument('--train_size_per_gpu', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='Llama-32-3B-Instruct')
    parser.add_argument('--save_path', type=str, default='llama_32_3b_instruct_lapt')
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--mode', choices=['lapt'], default='lapt')
    args = parser.parse_args()    
    print('Mode:', args.mode)
    # rewrite save path
    args.save_path += '_' + args.mode
    
    main(args)