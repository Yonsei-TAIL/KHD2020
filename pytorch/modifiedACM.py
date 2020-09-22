import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class modifiedACM(nn.Module):

    def __init__(self, model, num_heads, num_features, num_class=4, orthogonal_loss=True):
        super(modifiedACM, self).__init__()

        assert num_features % num_heads == 0

        self.model = model(self.num_features, num_heads=num_heads)

        self.out = num_class

        self.num_features = num_features
        self.num_heads = num_heads

        self.orthogonal_loss = orthogonal_loss

        self.init_parameters()

        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(num_heads, self.out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out, momentum = 0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, r, l):

        # creates left feature
        left = self.model(r)  # L
        left_feature = self.GAP(left)

        # creates right feature
        right = self.model(l)  # R
        right_feature = self.GAP(right)

        # difference modeling
        sub = left_feature - right_feature

        left = self.conv(left + sub)
        right = self.conv(right + sub)

        orth_loss = torch.mean(right_feature * left_feature, dim=1, keepdim=True)

        return left, right, orth_loss

    def init_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)