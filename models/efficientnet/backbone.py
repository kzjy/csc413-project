import torch
import torchvision.models as models

class EfficientNet(torch.nn.Module):

    def __init__(self):
        super(EfficientNet, self).__init__()
        self.net = models.efficientnet_b0(pretrained=True)
    
    def forward(self, input):
        return self.net(input)