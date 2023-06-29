
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegHRNet(nn.Module):
    def __init__(self, classes, pretrained=True, cfg='hrnet_w18_small_v2'):
        super().__init__()
        self.feature_extractor = timm.create_model(
            cfg, 
            pretrained=pretrained, 
            features_only=True
        )

        BatchNorm2d = torch.nn.BatchNorm2d # torch.nn.SyncBatchNorm
        relu_inplace = True
        BN_MOMENTUM = 0.1
        NUM_CLASSES = classes

        last_inp_channels = sum([64, 128, 256, 512])

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=1 if 1 == 3 else 0)
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        # Upsampling
        ALIGN_CORNERS = None
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x