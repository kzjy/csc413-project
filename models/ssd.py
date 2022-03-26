import torch
from backbone import EfficientNet

class SSD(torch.nn.Module):

    def __init__(self):
        super(SSD, self).__init__()
        self.backbone = EfficientNet()
    
    def forward(self, input):
        return self.backbone(input)