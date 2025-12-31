'''
Train a de-dimension network
'''
import torch
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def split_train_test_data(data, labels):
    num_samples = len(data)
    pos_data = [data[i] for i in range(num_samples) if labels[i] == 'pos']
    neg_data = [data[i] for i in range(num_samples) if labels[i] == 'neg']
    train_data = pos_data[:int(0.8 * len(pos_data))] + neg_data[:int(0.8 * len(neg_data))]
    test_data = pos_data[int(0.8 * len(pos_data)):] + neg_data[int(0.8 * len(neg_data)):]
    train_labels = ['pos'] * int(0.8 * len(pos_data)) + ['neg'] * int(0.8 * len(neg_data))
    test_labels = ['pos'] * (len(pos_data) - int(0.8 * len(pos_data))) + ['neg'] * (len(neg_data) - int(0.8 * len(neg_data)))
    return train_data, test_data, train_labels, test_labels

class PCAModel:
    
    def __init__(self, model, tokenizer, num_components=2, data=None, labels=None):
        if data is not None and labels is not None:
            self.train_data, self.test_data, self.train_labels, self.test_labels = split_train_test_data(data, labels)
        self.model = model
        self.tokenizer = tokenizer
        self.num_components = num_components
      
    def create_hidden_states(self, data, mode='sequential_w_pad'):
        '''
        There are three modes here: sequential_w_pad, sequential_wo_pad, batch_w_pad, batch_wo_pad
        '''
        if not isinstance(data, list):
            data = [data]
        num_layers = self.model.config.num_hidden_layers + 1
        num_hidden_size = self.model.config.hidden_size
        all_layers_last_hidden = torch.zeros((num_layers, len(data), num_hidden_size), device=self.model.device)
        if mode == 'sequential_w_pad':
            for i in range(len(data)):
                inputs = self.tokenizer(data[i], return_tensors="pt", padding=True, truncation=True, max_length=100).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    for j in range(num_layers):
                        hidden_states = outputs.hidden_states[j][:, -1, :].squeeze(1)
                        all_layers_last_hidden[j, i, :] = hidden_states
        elif mode == 'sequential_wo_pad':
            for i in range(len(data)):
                inputs = self.tokenizer(data[i], return_tensors="pt", padding=False, truncation=True, max_length=100).to(self.model.device)
                seq_len = inputs['attention_mask'].sum().item() - 1
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    for j in range(num_layers):
                        hidden_states = outputs.hidden_states[j][:, seq_len, :].squeeze(1)
                        all_layers_last_hidden[j, i, :] = hidden_states
        elif mode == 'batch_w_pad':
            inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=100).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                for j in range(num_layers):
                    hidden_states = outputs.hidden_states[j][:, -1, :].squeeze(1)
                    all_layers_last_hidden[j, :, :] = hidden_states
        elif mode == 'batch_wo_pad':   
            inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=100).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                total_hidden_states = outputs.hidden_states  # tuple of tensors
            attention_mask = inputs['attention_mask']
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = attention_mask.size(0)
            batch_indices = torch.arange(batch_size, device=self.model.device)
            for i, layer_hidden_states in enumerate(total_hidden_states):
                all_layers_last_hidden[i] = layer_hidden_states[batch_indices, seq_lengths]
        return all_layers_last_hidden
    
    def token_compare(self):
        cnt = 0
        for data in self.train_data:
            prompt_inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=100).to(self.model.device)
            # 统计padding的
            prompt_len = len(prompt_inputs['attention_mask']) -  prompt_inputs['attention_mask'].sum().item()
            if prompt_len > 0:
                cnt += 1
        print(f'Number of padding: {cnt}')
        
    def export_pca_to_csv(self):
        '''
        Export PCA points to CSV
        '''
        mode = 'batch_wo_pad'
        train_hidden_states = self.create_hidden_states(self.train_data, mode)
        pca = PCA(n_components=self.num_components)
        hidden_states_for_layer_i = train_hidden_states[26].cpu().detach().numpy()
        reduced_hidden_states = pca.fit_transform(hidden_states_for_layer_i)
        reduced_hidden_states = reduced_hidden_states.tolist()
        # x, y, label
        reduced_hidden_states = [[x[0], x[1], y] for x, y in zip(reduced_hidden_states, self.train_labels)]
        with open(f'results/{mode}_pca_layer_26.csv', 'w') as f:
            f.write('x,y,label\n')
            for item in reduced_hidden_states:
                f.write(f'{item[0]},{item[1]},{item[2]}\n')
        print(f'Export PCA to CSV successfully')
        
    def train(self):
        mode = 'batch_wo_pad'
        train_hidden_states = self.create_hidden_states(self.train_data, mode)
        for i in tqdm(range(self.model.config.num_hidden_layers + 1) ):
            pca = PCA(n_components=self.num_components)
            hidden_states_for_layer_i = train_hidden_states[i].cpu().detach().numpy()
            reduced_hidden_states = pca.fit_transform(hidden_states_for_layer_i)
            plot_distribution(reduced_hidden_states, self.train_labels, f'results/{mode}_pca_layer_{i}.png')
            joblib.dump(pca, f'pca/{mode}_pca_{i}.pkl')
    
    def test(self):
        mode = 'batch_wo_pad'
        test_hidden_states = self.create_hidden_states(self.test_data)
        for i in range(self.model.config.num_hidden_layers + 1):
            # gather all hidden states of layer i
            hidden_states_for_layer_i = test_hidden_states[i].cpu().detach().numpy()
            pca = joblib.load(f'pca/{mode}_pca_{i}.pkl')
            reduced_hidden_states = pca.transform(hidden_states_for_layer_i)
            plot_distribution(reduced_hidden_states, self.test_labels, f'test_pca_layer_{i}.png')   
            print(f'Plot distribution for layer {i} successfully')
    
