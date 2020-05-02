import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentNet(nn.Module):
    def __init__(self, base_channel_size=16, width_mult=1):
        '''
          Args:
            base_channel_size: 這個model一開始的ch數量，每過一層都會*2，直到base_channel_size*16為止。
            width_mult: 為了之後的Network Pruning使用，在base_channel_size*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [ base_channel_size * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        layers = [self.conv_block(3, bandwidth[0], 3, 1, 1, max_pooling=True)]
        layers.extend([self.conv_block(in_ch, out_ch, 3, 1, 1, groups=in_ch, max_pooling=(i < 3)) for i, (in_ch, out_ch) in enumerate(zip(bandwidth[:-1], bandwidth[1:]))])
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(bandwidth[7], 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, max_pooling=True):
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels, out_channels, 1)
            ]
        if max_pooling:
            layers.append(nn.MaxPool2d(2, 2, 0))
        return nn.Sequential(*layers)
