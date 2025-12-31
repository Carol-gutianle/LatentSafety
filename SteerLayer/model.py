import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from sv.utils import cos_similarity, prepare_data
from sv.pca import PCAModel

class SteerLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    def forward(self, hidden_states):
        steer_vector = self.mlp(hidden_states)
        return steer_vector
    
def save_checkpoint(model, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)
    
def train(args):

    def loss_fn(h, target):
        return torch.norm(h - target, p=2)
    
    def early_stopping(loss_before, loss_after, threshold=1e-3, patience=5):
        """
        Determines whether early stopping should occur based on loss improvement.

        :param loss_before: The loss value from the previous iteration.
        :param loss_after: The loss value from the current iteration.
        :param threshold: The minimum change in loss between iterations to continue training.
        :param patience: Number of iterations with insufficient improvement before stopping.
        :return: True if early stopping criteria are met, False otherwise.
        """
        if abs(loss_after - loss_before) < threshold:
            patience -= 1
            if patience <= 0:
                return True, patience
        else:
            patience = 5
        return False, patience

            
    pos_data, neg_data = prepare_data(
        'alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet',
        'advbench/harmful_strings.csv',
        100
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code = True,
        output_hidden_states = True,
        torch_dtype = 'auto',
        device_map = 'auto'
    ).eval()
    
    tokenzier = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )

    pca = PCAModel(model, tokenzier)
    
    query_prompt = [neg_data[0]]
    
    query_states = pca.create_hidden_states(query_prompt)[args.layer_idx]
    safe_states = pca.create_hidden_states(pos_data)[args.layer_idx]
    unsafe_states = pca.create_hidden_states(neg_data)[args.layer_idx]
    
    safe_center = torch.mean(safe_states, axis=0)
    unsafe_center = torch.mean(unsafe_states, axis=0)
    
    # initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    steer_vector = SteerLayer(args.hidden_dim).to(device)
    
    optimizer = optim.Adam(steer_vector.mlp.parameters(), lr=args.lr)
    
    epoch = 0
    loss_before = 0
    loss_after = 0
    patience = 1
    
    while True:
        
        epoch += 1
        print(f"Epoch {epoch}")
        
        optimizer.zero_grad()
        loss_contrast = loss_fn(steer_vector(query_states) + query_states, unsafe_center)
        loss = loss_contrast + args.alpha * (1 - cos_similarity(steer_vector(query_states) + query_states, query_states))
        
        loss.backward()
        optimizer.step()
        
        loss_after = loss.item()
        stop_flg, patience = early_stopping(loss_before, loss_after, 1e-1, patience)
        
        if stop_flg:
            save_checkpoint(steer_vector, epoch, f"{args.steer_layer_save_path}")
            print(f"Early stopping at epoch {epoch}")
            print(f"Loss after: {loss_after}")
            break
        else:
            loss_before = loss_after
            
            
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="SteerLayer")
    parser.add_argument("--model_name_or_path", type=str, default="models--Qwen--Qwen2.5-7B-Instruct", help="Pretrained model name or path")
    parser.add_argument("--steer_layer_save_path", type=str, default="SteerLayer/steer_layer.pt", help="Steer layer save path")
    parser.add_argument("--layer_idx", type=int, default=12, help="Layer index")
    parser.add_argument("--hidden_dim", type=int, default=3584, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1e-2, help="Regularization coefficient")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train(args)
