import os
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sv.intervent import SteerVectorVisual
from sv.utils import get_model_and_tokenizer, set_seed, read_data_from_json, fetch_pos_neg_data

class ActivationJail:
    
    def __init__(self, prompts, suffix_list, method, save_path, model_name_or_path, max_new_tokens, seed, pos_data=None, neg_data=None):
        '''
        prompts: list of prompts to jailbreak
        method: method to jailbreak
        save_path: path to save the results (json)
        '''
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = get_model_and_tokenizer(model_name_or_path)
        self.prompts = prompts
        self.method = method
        self.suffix_list = suffix_list
        if self.method == 'gasa' and len(suffix_list) == 0:
            raise ValueError('You must provide suffix_list when using ASAgrad!')
        self.save_path = save_path
        self.max_new_tokens = max_new_tokens
        self.interventor = SteerVectorVisual(self.model, self.tokenizer, seed, pos_data, neg_data)
        
    def jailbreak(self):
        num_layers = self.model.config.num_hidden_layers
        
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'type': 'configuration',
            'model': self.model_name_or_path,
            'method': self.method,
            'total_layers': num_layers,
            'total_prompts': len(self.prompts)
        }
        
        with open(self.save_path, 'w') as f:
            f.write(json.dumps(config) + '\n')
        
        prompt_bar = tqdm(enumerate(self.prompts), total=len(self.prompts), desc="Processing prompts")
        
        for prompt_idx, prompt in prompt_bar:
            layer_bar = tqdm(range(num_layers), total=num_layers, 
                            desc=f"Prompt {prompt_idx+1}/{len(self.prompts)}", 
                            leave=False)
            
            for layer_idx in layer_bar:
                if self.method == 'gasa':
                    target_suffix = self.suffix_list[prompt_idx]
                else:
                    target_suffix = None
                original_response, steered_response = self.interventor.generate_w_steer(
                    prompt,
                    [layer_idx],
                    self.max_new_tokens,
                    self.method,
                    target_suffix = target_suffix
                )
                result = {
                    'type': 'result',
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'suffix': self.suffix_list[prompt_idx] if len(self.suffix_list) != 0 else None,
                    'layer_idx': layer_idx,
                    'original_response': original_response,
                    'steered_response': steered_response
                }
                with open(self.save_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                
                layer_bar.set_description(f"Layer {layer_idx+1}/{num_layers}")
            
            prompt_bar.set_description(f"Completed prompt {prompt_idx+1}/{len(self.prompts)}")
        
        print(f"All results saved to {self.save_path}")
        
        return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='DeepSeek-R1-Distill-Qwen-1.5B', help='Path to the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--jailbreak', type=str, default='random', help='Jailbreak method to use')
    parser.add_argument('--dataset', type=str, default='advbench', help='Dataset to use')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Max new tokens to generate')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    args = parser.parse_args()
    set_seed(args.seed)
    path_to_name = {
        'Llama-3.2-3B': 'Llama-32-3B-Base',
        'Llama-3.2-3B-Instruct': 'Llama-32-3B-Instruct',
        'Qwen2.5-7B': 'Qwen-25-7B-Base',
        'Qwen2.5-7B-Instruct': 'Qwen-25-7B-Instruct',
        'Llama-3.1-8B': 'Llama-31-8B-Base',
        'Llama-3.1-8B-Instruct': 'Llama-31-8B-Instruct',
    }
    # global settings
    model_name = os.path.basename(args.model_name_or_path)
    if args.dataset == 'advbench':
        prompt_path = '../data/advbench_harmful_behaviors.json'
        # load prompts 
        prompts = read_data_from_json(prompt_path)
    elif args.dataset == 'gcg':
        prompt_path = f'../data/gcg/{path_to_name[args.model_name_or_path]}.csv'
    elif args.dataset == 'phihex':
        pass
    if args.dataset == 'gcg':
        prompts = pd.read_csv(prompt_path).to_dict(orient='records')
        seed_prompts = []
        for prompt in prompts:
            if not pd.isna(prompt['best_suffix']):
                seed_prompts.append(prompt['goal'] + prompt['best_suffix'])
    else:
        seed_prompts = [prompt['goal']  for prompt in prompts][:args.num_samples]
    if args.jailbreak == 'gasa':
        # Note: for ablation
        suffix_list = [prompt['target'] + '!' for prompt in prompts]
    else:
        suffix_list = []
    if args.jailbreak == 'trojan':
        # get pos_data and neg_data (each for 50 samples)
        pos_data, neg_data = fetch_pos_neg_data(50)
    else:
        pos_data = None
        neg_data = None
    # jailbreak
    jailbreak = ActivationJail(
        seed_prompts, 
        suffix_list, 
        args.jailbreak,
        '{args.seed}_{args.jailbreak}_{args.max_new_tokens}_{model_name}_{args.num_samples}.json',
        args.model_name_or_path,
        args.max_new_tokens,
        args.seed,
        pos_data,
        neg_data
    )
    jailbreak.jailbreak()