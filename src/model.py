import torch
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F

class CrowdCounterNet(nn.Module):
    def __init__(self, pretrained=True):

        super(CrowdCounterNet, self).__init__() 

        vgg = vgg19(weights='VGG19_Weights.DEFAULT' if pretrained else None)

        self.backbone = vgg.features[:35]

        self.regression_header = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        )

    def forward(self, x):

        x = self.backbone(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        density_map = self.regression_header(x)
        return density_map