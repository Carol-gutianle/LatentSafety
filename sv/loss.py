'''
Scripts used to calculate loss
'''
import json
import torch
from tqdm import tqdm
from sv.utils import set_seed, get_model_and_tokenizer, read_data_from_jsonlines
from sv.visualize import load_advbench_target
  

def fetch_layers_by_ids(model, layer_list):
    '''
    Fetch layers by ids
    '''
    target_layers = []
    if hasattr(model, 'transformer'):
        for layer_idx in layer_list:
            if layer_idx < 0:
                layer_idx = len(model.transformer.h) + layer_idx
            target_layers.append(model.transformer.h[layer_idx])
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            for layer_idx in layer_list:
                if layer_idx < 0:
                    layer_idx = len(model.model.layers) + layer_idx
                target_layers.append(model.model.layers[layer_idx])
        elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            for layer_idx in layer_list:
                if layer_idx < 0:
                    layer_idx = len(model.model.decoder.layers) + layer_idx
                target_layers.append(model.model.decoder.layers[layer_idx])
        else:
            raise ValueError(f"Model Not Implemented: {model.__class__.__name__}")
    else:
        raise ValueError(f"Model Not Implemented: {model.__class__.__name__}")
    return target_layers

class LossAnalyzer:
    
    def __init__(self, model, tokenizer, seed=42):
        self.seed = seed
        self.model = model
        self.tokenizer = tokenizer
        
    def gasa_attack(self, seed_prompt, target_suffix, layer_list, normal_hidden_states):
        self.model.train()
        full_prompt = seed_prompt[0] + target_suffix
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs['input_ids']
        target_layers = fetch_layers_by_ids(self.model, layer_list)
        for target_layer in target_layers:
            activation_store = {}
            def hook_fn(module, input, output):
                output[0].retain_grad()
                activation_store['act'] = output[0]
                return output
            hook = target_layer.register_forward_hook(hook_fn)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer(seed_prompt)['input_ids'][0])
        labels[:, :prompt_len] = -100
        inputs['labels'] = labels
        output = self.model(**inputs)
        self.model.eval()
        loss = output.loss
        loss.backward()
        hook.remove()
        epsilon = 1
        delta = epsilon * activation_store['act'].grad.sign()
        steer_vector = delta[:, -2, :].detach()
        steer_vector = steer_vector.to(normal_hidden_states.device)
        steer_vector = self.normalize_delta_to_preserve_stats(normal_hidden_states, steer_vector)
        return steer_vector
        
    def get_static_steer_vector(self, normal_hidden_states, alpha=1):
        '''
        Create random static steer_vector
        '''
        set_seed(self.seed)
        random_center = torch.randn_like(normal_hidden_states)
        random_center = random_center * alpha
        random_center = random_center.to(normal_hidden_states.device)
        random_center = self.normalize_delta_to_preserve_stats(normal_hidden_states, random_center)
        return random_center
    
    def normalize_delta_to_preserve_stats(self, reference, delta):
        """
        Normalize each row in delta to have the same mean and std as the corresponding row in reference.

        Args:
            reference: Tensor of shape (num_samples, hidden_dim)
            delta: Tensor of shape (num_samples, hidden_dim)

        Returns:
            norm_delta: Tensor of shape (num_samples, hidden_dim)
        """
        mean_r = reference.mean(dim=1, keepdim=True) 
        std_r = reference.std(dim=1, keepdim=True)   

        mean_d = delta.mean(dim=1, keepdim=True)
        std_d = delta.std(dim=1, keepdim=True)

        norm_delta = (delta - mean_d) / (std_d + 1e-6) * std_r + mean_r
        return norm_delta
    
    
    def loss_analyze(self, seed_prompt, prefix_prompt, mode, layer_list, alpha, beta, target_suffix=None):
        '''
        Used for loss analysis
        '''
        prompt_input_ids = self.tokenizer(prefix_prompt, return_tensors="pt").input_ids
        prompt_len = prompt_input_ids.size(1)
        steer_vector_dict = {}
        num_layer = self.model.config.num_hidden_layers + 1
        if not isinstance(seed_prompt, list):
            seed_prompt = [seed_prompt]
        num_samples = len(seed_prompt)
        normal_hidden_states = torch.zeros((num_layer, num_samples, self.model.config.hidden_size), device=self.model.device)
        # normal inference
        inputs = self.tokenizer(seed_prompt, return_tensors="pt").to(self.model.device)
        inputs_seq_len = inputs["attention_mask"].sum(dim=-1).sub_(1)
        batch_size = inputs["input_ids"].size(0)
        batch_indices = torch.arange(batch_size)
        with torch.no_grad():
            origin_outputs = self.model(**inputs, labels=inputs["input_ids"], output_hidden_states=True, do_sample=False)
            origin_outputs_hidden_states = origin_outputs.hidden_states
            origin_loss = origin_outputs.loss
        for i in range(num_layer):
            normal_hidden_states[i] = origin_outputs_hidden_states[i][batch_indices, inputs_seq_len]
        for layer_idx in layer_list:
            if mode == 'random':
                return_static_vector = self.get_static_steer_vector(normal_hidden_states[layer_idx])
                steer_vector = return_static_vector
                steer_vector_dict[layer_idx] = steer_vector
            elif mode == 'gasa':
                if target_suffix is None:
                    raise ValueError('Suffix List should not be None!')
                steer_vector = self.fgsm_attack(seed_prompt, target_suffix, layer_list, normal_hidden_states[layer_idx])
                steer_vector_dict[layer_idx] = steer_vector
            else:
                raise ValueError(f"Mode {mode} not supported!")
        # register hooks
        hooks = []
        def create_pos_steering_hook(layer_idx):
            def hook_fn(module, input, output):
                steer_vector = steer_vector_dict[layer_idx]
                if not hasattr(hook_fn, 'applied') or hook_fn.applied == False:
                    if isinstance(output, tuple):
                        modified = output[0].clone()
                        modified[:, prompt_len - 1 : -1, :] += steer_vector.to(modified.device)
                        new_output = (modified,) + output[1:] if len(output) > 1 else (modified,)
                        return new_output
                    else:
                        modified = output.clone()
                        modified[:, -1, :] += steer_vector.to(modified.device)
                        return modified
                return output
            hook_fn.applied = False
            return hook_fn
        for i, target_layer in enumerate(fetch_layers_by_ids(self.model, layer_list)):
            layer_idx = layer_list[i]
            hook = create_pos_steering_hook(layer_idx)
            hooks.append(target_layer.register_forward_hook(hook))
        with torch.no_grad():
            steered_outputs = self.model(**inputs, labels=inputs["input_ids"], output_hidden_states=True, do_sample=False)
            steered_loss = steered_outputs.loss
        for hook in hooks:
            hook.remove()
        
        return origin_loss, steered_loss
    
        
if __name__ == "__main__":
    '''
    Experiments with LossAnalyzer
    {
        'prompt_idx':,
        'prompt':,
        'layer_idx':,
        'original_response':,
        'steered_response':,
        'model_name':,
        'target_suffix':,
        'losses': {
            'random': {
                'clean_loss_to_origin_response': float,
                'clean_loss_to_steered_response': float,
                'steered_loss_to_origin_response': float,
                'steered_loss_to_steered_response': float
            },
            'fgsm': {
                'clean_loss_to_origin_response': float,
                'clean_loss_to_steered_response': float,
                'steered_loss_to_origin_response': float,
                'steered_loss_to_steered_response': float
            }
        }
    }
    '''
    model_path_dict = {
        'Llama-32-3B': 'Llama-3.2-3B',
        'Llama-32-3B-Instruct': 'Llama-3.2-3B-Instruct',
        'Qwen-25-7B': 'Qwen2.5-7B',
        'Qwen-25-7B-Instruct': 'Qwen2.5-7B-Instruct',
        'Llama-31-8B': 'Llama-3.1-8B',
        'Llama-31-8B-Instruct': 'Llama-3.1-8B-Instruct',
        'Llama-2-13B-Chat': 'Llama-2-13b-chat-hf',
    }
    data_path_dict = {
        'Llama-32-3B': '42_random_50_Llama-3.2-3B_100_0_100.json',
        'Llama-32-3B-Instruct': '42_random_50_Llama-3.2-3B-Instruct_100_0_100.json',
        'Qwen-25-7B': '42_random_50_Qwen2.5-7B_100_0_100.json',
        'Qwen-25-7B-Instruct': '42_random_50_Qwen2.5-7B-Instruct_100_0_100.json',
        'Llama-31-8B': '42_random_50_Llama-3.1-8B_100_0_100.json',
        'Llama-31-8B-Instruct': '42_random_50_Llama-3.1-8B-Instruct_100_0_100.json',
        'Llama-2-13B-Chat': 'Llama-2-13B-Chat_asabench_seed42.json',
    }
    model_name = 'Qwen-25-7B'
    data = read_data_from_jsonlines(data_path_dict[model_name])
    model, tokenizer = get_model_and_tokenizer(model_path_dict[model_name])
    analyzer = LossAnalyzer(model, tokenizer)
    adv_target = load_advbench_target()
    # dataframe
    output_path = f'{model_name}.jsonl'

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(data):
            if 'prompt' not in sample:
                continue
            prefix_prompt = sample['prompt']
            original_response = sample['original_response']
            steered_response = sample['steered_response']

            # find target suffix via prompt_idx == id in adv_df
            prompt_idx = sample['prompt_idx']
            target_suffix = adv_target[int(prompt_idx)]
            layer_list = sample['layer_idx']

            # loss analysis
            methods = ['random', 'fgsm']
            sample['losses'] = {}

            for method in methods:
                clean_loss_to_origin_response, noisy_loss_to_origin_response = analyzer.loss_analyze(
                    prefix_prompt + original_response,
                    prefix_prompt,
                    method,
                    [layer_list],
                    target_suffix=target_suffix
                )
                clean_loss_to_steered_response, noisy_loss_to_steered_response = analyzer.loss_analyze(
                    prefix_prompt + steered_response,
                    prefix_prompt,
                    method,
                    [layer_list],
                    target_suffix=target_suffix
                )
                sample['losses'][method] = {
                    'clean_loss_to_origin_response': clean_loss_to_origin_response.item(),
                    'noisy_loss_to_origin_response': noisy_loss_to_origin_response.item(),
                    'clean_loss_to_steered_response': clean_loss_to_steered_response.item(),
                    'noisy_loss_to_steered_response': noisy_loss_to_steered_response.item()
                }

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            