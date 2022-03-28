import torch
from models.backbone import EfficientNet

class EfficientDet(torch.nn.Module):

    def __init__(self):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet()
    
    def forward(self, input):
        return self.backbone(input)