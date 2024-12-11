import torch.nn as nn
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    torch.set_default_device('mps')
    device = torch.device('mps')
else:
    torch.set_default_device('cpu')
    device = torch.device('cpu')

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = x.to(device)
        x = resnet(x).to(device)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.sigmoid(self.fc1(x))
        x = torch.flatten(x, 0)
        return x