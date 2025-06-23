import torch
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from sv.utils import set_seed

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

class SteerVectorVisual:
    
    def __init__(self, model, tokenizer, seed=42, pos_data=None, neg_data=None):
        self.seed = seed
        self.model = model
        self.tokenizer = tokenizer
        self.pos_data = pos_data
        self.neg_data = neg_data
        
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
        # normalize
        steer_vector = steer_vector.to(normal_hidden_states.device)
        steer_vector = self.normalize_delta_to_preserve_stats(normal_hidden_states, steer_vector)
        # steer_vector (num_samples, hidden_states)
        return steer_vector
    
        
    def get_static_steer_vector(self, normal_hidden_states, alpha=1):
        '''
        Create random static steer_vector
        '''
        # Very Important!
        set_seed(self.seed)
        random_center = torch.randn_like(normal_hidden_states)
        random_center = random_center * alpha
        random_center = random_center.to(normal_hidden_states.device)
        random_center = self.normalize_delta_to_preserve_stats(normal_hidden_states, random_center)
        return random_center
    
    def trojan_attack(self, positive_prompts, negative_prompts, normal_hidden_states, layer_idx):
        '''
        Create steer vector, from positive_prompts to negative_prompts
        '''
        from sv.pca import PCAModel
        pca = PCAModel(self.model, self.tokenizer)
        pos_hidden_states = pca.create_hidden_states(positive_prompts)
        neg_hidden_states = pca.create_hidden_states(negative_prompts)
        pos_center = torch.mean(pos_hidden_states, axis = 1)
        neg_center = torch.mean(neg_hidden_states, axis = 1)
        steer_center = neg_center - pos_center
        steer_center[layer_idx] = self.normalize_delta_to_preserve_stats(normal_hidden_states, steer_center[layer_idx])
        return steer_center
    
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
    
    
    def generate_w_steer(self, seed_prompt, layer_list, max_new_tokens, mode='mlp', verbose=False, export_hidden_states=False, target_suffix=None):
        '''
        mode used to ablation study.
        1. mlp: use mlp to steer the hidden states
        2. static: use negative data hidden states center as the steered vector
        '''
        from SteerLayer.model import SteerLayer
        # prepare for hidden states
        if export_hidden_states:
            origin_hidden_states = {layer_list[i]: [] for i in range(len(layer_list))}
            steered_hidden_states = {layer_list[i]: [] for i in range(len(layer_list))}
        checkpoint = torch.load('steer_layer.pt')
        steer_model = SteerLayer(self.model.config.hidden_size).to(self.model.device)
        if mode == 'mlp':
            steer_model.load_state_dict(checkpoint['model_state_dict'])
            steer_model.eval()
        else:
            steer_model = None
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
            origin_outputs = self.model(**inputs, do_sample=False)
            origin_outputs_hidden_states = origin_outputs.hidden_states
            origin_generations = self.model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
            origin_logits = origin_generations.scores
            origin_output_ids = origin_generations.sequences
            origin_prompts = self.tokenizer.decode(origin_output_ids[0], skip_special_tokens=True).replace(seed_prompt[0], '')
        for i in range(num_layer):
            normal_hidden_states[i] = origin_outputs_hidden_states[i][batch_indices, inputs_seq_len]
        for layer_idx in layer_list:
            if mode == 'mlp':
                steer_vector_dict[layer_idx] = steer_model(normal_hidden_states[layer_idx])
            elif mode == 'random':
                return_static_vector = self.get_static_steer_vector(normal_hidden_states[layer_idx])
                steer_vector = return_static_vector
                steer_vector_dict[layer_idx] = steer_vector
            elif mode == 'gasa':
                if target_suffix is None:
                    raise ValueError('Suffix List should not be None!')
                steer_vector = self.fgsm_attack(seed_prompt, target_suffix, layer_list, normal_hidden_states[layer_idx])
                steer_vector_dict[layer_idx] = steer_vector
            elif mode == 'trojan':
                # duplicate the negative data hidden states * num_samples
                return_static_vector = self.trojan_attack(self.pos_data, self.neg_data, normal_hidden_states[layer_idx], layer_idx).squeeze(0)
                layer_static_vector = return_static_vector[layer_idx]
                steer_vector = torch.cat([layer_static_vector for _ in range(num_samples)], dim=0)
                steer_vector_dict[layer_idx] = steer_vector
            else:
                raise NotImplementedError(f"Mode {mode} Not Implemented")
        # register hooks
        hooks = []
        def create_pos_steering_hook(layer_idx):
            def hook_fn(module, input, output):
                steer_vector = steer_vector_dict[layer_idx]
                if not hasattr(hook_fn, 'applied') or hook_fn.applied == False:
                    if isinstance(output, tuple):
                        modified = output[0].clone()
                        # add steer vector to the hidden states of last token
                        modified[:, -1, :] += steer_vector.to(modified.device)
                        new_output = (modified,) + output[1:] if len(output) > 1 else (modified,)
                        if export_hidden_states:
                            origin_hidden_states[layer_idx].append(output[0][:, -1, :].clone())
                            steered_hidden_states[layer_idx].append(modified[:, -1, :].clone())
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
            steered_generations = self.model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
            output_ids = steered_generations.sequences
            steered_logits = steered_generations.scores
            steered_prompts = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(seed_prompt[0], '')
        for hook in hooks:
            hook.remove()
        
        if verbose:
            console = Console()
            
            origin_text = Text(origin_prompts)
            origin_panel = Panel(
                origin_text,
                title=f"[bold]Orgin Output (Layer {layer_list})",
                border_style="green"
            )
            
            steered_text = Text(steered_prompts)
            steered_panel = Panel(
                steered_text,
                title=f"[bold]Steered Output (Layer {layer_list})",
                border_style="red"
            )
            
            console.print("\nPrompt:", seed_prompt[0])
            console.print(origin_panel)
            console.print(steered_panel)
        if export_hidden_states:
            return origin_prompts, steered_prompts, origin_hidden_states, steered_hidden_states, origin_logits, steered_logits
        else:
            return origin_prompts, steered_prompts
        