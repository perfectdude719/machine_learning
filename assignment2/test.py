import torch
print(torch.cuda.is_available())  # Should print True if the GPU is ready
print(torch.cuda.device_count())  # Should return the number of GPUs detected
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
