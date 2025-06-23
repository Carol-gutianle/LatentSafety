'''
Scripts for landscape analysis.
'''
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sv.visualize import load_advbench_target
from sv.utils import get_model_and_tokenizer, read_data_from_json
from sv.loss import LossAnalyzer
from matplotlib import cm, colors
from matplotlib.colors import LightSource

def plot_landscape(model_name_or_path, data_path):
    # load_data
    data = read_data_from_json(data_path)
    model, tokenizer = get_model_and_tokenizer(model_name_or_path)
    loss_analyzer = LossAnalyzer(model, tokenizer)
    
    # for gasa
    alpha = np.linspace(0.0, 1.0, 50)
    # for random
    beta = np.linspace(0.0, 1.0, 50)
    alpha_grid, beta_grid = np.meshgrid(alpha, beta)
    
    Z_all = []
    adv_target = load_advbench_target()

    data = data[:20]
    
    for prompt in tqdm(data, leave=False):
        Z_prompt = np.zeros_like(alpha_grid)
        for i in tqdm(range(alpha_grid.shape[0]), leave=False):
            for j in tqdm(range(alpha_grid.shape[1]), leave=False):
                alpha = float(alpha_grid[i, j])
                beta = float(beta_grid[i, j])
                _, loss_val = loss_analyzer.loss_analyze(
                    seed_prompt = prompt['prompt'] + prompt['original_response'],
                    prefix_prompt = prompt['prompt'],
                    mode = 'combine',
                    alpha = alpha,
                    beta = beta,
                    layer_list = [prompt['layer_idx']],
                    target_suffix = adv_target[int(prompt['prompt_idx'])]
                )
                Z_prompt[i, j] = loss_val
        Z_all.append(Z_prompt)
    

    Z_avg = np.mean(np.stack(Z_all, axis=0), axis=0)
    np.save("landscape.npy", Z_avg)
    
    Z_avg = np.load("landscape.npy")
    
    ls = LightSource(azdeg=135, altdeg=45)
    rgb = ls.shade(Z_avg, cmap=cm.coolwarm, vert_exag=2.0, blend_mode='soft')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(beta_grid, alpha_grid, Z_avg, facecolors=rgb, linewidth=0.2, antialiased=True, shade=False)
    ax.plot_wireframe(
        beta_grid, alpha_grid, Z_avg,
        color='gray',
        linewidth=0.2,
        alpha=0.4
    )
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    ax.contourf(beta_grid, alpha_grid, Z_avg, zdir='z', offset=Z_avg.min(), cmap='coolwarm', alpha=0.6)

    ax.set_ylabel("GASA (β)", fontsize=14)
    ax.set_xlabel("Random (γ)", fontsize=14)
    
    norm = colors.Normalize(vmin=Z_avg.min(), vmax=Z_avg.max())
    mappable = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    mappable.set_array([])
    
    min_idx = np.unravel_index(np.argmin(Z_avg), Z_avg.shape)
    min_alpha = alpha_grid[min_idx]
    min_beta = beta_grid[min_idx]
    min_val = Z_avg[min_idx]
        
    ax.scatter(
        min_beta, min_alpha, min_val, 
        color='black', s=50, marker='o', label='min'
    )

    ax.text(
        min_beta, min_alpha, min_val,
        f'({min_alpha:.2f}, {min_beta:.2f})',
        fontsize=14, color='black'
    )

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)

    cbar.set_label('NLL to original response', fontsize=14)

    cbar.ax.tick_params(labelsize=14)    

    plt.tight_layout()
    plt.savefig('landscape.pdf', dpi=300)
    
plot_landscape('Llama-3.2-3B', 'Llama-32-3B_asabench_seed42.json')