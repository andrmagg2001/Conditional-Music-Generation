import torch

path = "data/checkpoints/baseline_transformer/best.pth"
data = torch.load(path, map_location='cpu', weights_only=False)


state_dict = data['model'] if 'model' in data else data

total_params = 0

for name, param in state_dict.items():
    if isinstance(param, torch.Tensor):
        total_params += param.numel()

print(f"--- Riepilogo Modello ---")
print(f"Parametri totali: {total_params:,}")
print(f"Dimensione stimata su disco: {total_params * 4 / (1024**2):.2f} MB (se float32)")