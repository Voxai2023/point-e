import torch
torch.zeros(1).cuda()
print(torch.cuda.is_available())