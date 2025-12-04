import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Unet(nn.Module):

    def __init__(self, num_classes: int=5, pretrained=True) -> None:
        super().__init__()
        
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        feature = backbone.features

        self.stem = feature[0]
        self.block1 = feature[1]
        self.block2 = feature[2]
        self.block3 = feature[3]
        self.block4 = feature[4]
        self.block5 = feature[5]
        self.block6 = feature[6]
        self.block7 = feature[7]

        self.up7 = nn.ConvTranspose2d(320, 192, 2, stride=2)
        self.dec7 = self.conv_block(192 + 192, 192)
        self.up6 = nn.ConvTranspose2d(192, 112, 2, stride=2)
        self.dec6 = self.conv_block(112 + 112, 112)
        self.up5 = nn.ConvTranspose2d(112, 80, kernel_size=2, stride=2)
        self.dec5 = self.conv_block(80 + 80, 80)
        self.up4 = nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(40 + 40, 40)
        self.up3 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(24 + 24, 24)
        self.up2 = nn.ConvTranspose2d(24, 16, kernel_size=1)
        self.dec2 = self.conv_block(16 + 16, 16)
        self.up1 = nn.ConvTranspose2d(16, 32, kernel_size=1)
        self.dec1 = self.conv_block(32 + 32, 32)

        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x0 = self.stem(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)

        x = self.up7(x7)
        x6 = self.resize(x6, x)
        x = self.dec7(torch.cat([x, x6], dim=1))

        x = self.up6(x)
        x5 = self.resize(x5, x)
        x = self.dec6(torch.cat([x, x5], dim=1))

        x = self.up5(x)
        x4 = self.resize(x4, x)
        x = self.dec5(torch.cat([x, x4], dim=1))

        x = self.up4(x)
        x3 = self.resize(x3, x)
        x = self.dec4(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x2 = self.resize(x2, x)
        x = self.dec3(torch.cat([x, x2], dim=1))

        x = self.up2(x)
        x1 = self.resize(x1, x)
        x = self.dec2(torch.cat([x, x1], dim=1))
        
        x = self.up1(x)
        x0 = self.resize(x0, x)
        x = self.dec1(torch.cat([x, x0], dim=1))

        logits = self.classifier(x)
        return logits

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def resize(self, source, target):
        return F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)