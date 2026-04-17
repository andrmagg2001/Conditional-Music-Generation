import torch

path = "data/checkpoints/baseline_transformer/best.pth"
data = torch.load(path, map_location='cpu', weights_only=False)


state_dict = data['model'] if 'model' in data else data

total_params = 0

for name, param in state_dict.items():
    if isinstance(param, torch.Tensor):
        total_params += param.numel()

print(f"--- Model Summary ---")
print(f"Total parameters: {total_params:,}")
print(f"Estimated disk size: {total_params * 4 / (1024**2):.2f} MB (if float32)")