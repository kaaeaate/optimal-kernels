import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_attention_map(attention_map, N=100, n_columns=5):
    normed_att_map = transforms.Normalize(0, 1)(attention_map).detach().cpu().numpy()
#     clear_output(wait=True)
    
    n_rows = N // n_columns + int(N // n_columns * n_columns < N)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15,6))
    for map_i in range(0, N, 10):
        if N==1:
            plt.imshow(normed_att_map[map_i])
        else:
            row_index = map_i // n_columns
            column_index = map_i % n_columns
            axes[row_index, column_index].imshow(normed_att_map[map_i])
    plt.show()
    
def get_feature_map(normed_att_map, index=3):
    normed_att_map = transforms.Normalize(0, 1)(attention_map).detach().cpu().numpy()
    plt.imshow(normed_att_map[index])
    plt.title('Choosen map')
    plt.show()