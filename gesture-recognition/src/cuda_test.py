import torch
import torch
print("torch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn available:", torch.backends.cudnn.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available() :
    print("You are now using cuda")
else :
    print("You are now using cpu")