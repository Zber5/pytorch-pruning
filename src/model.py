import torch.nn as nn
import torch.nn.functional as F

'''VGG11/13/16/19 in Pytorch.'''
import torch
from torchsummary import summary
from ptflops import get_model_complexity_info


class LeNet5_GROW_P(nn.Module):
    def __init__(self, out1=2, out2=5, fc1=10):
        super(LeNet5_GROW_P, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, out1, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out1, out2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(out2 * 16, fc1),
            nn.ReLU(inplace=True),
            nn.Linear(fc1, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out2


class LeNet5(nn.Module):
    def __init__(self, in_channel, out1_channel, out2_channel, fc, out_classes, kernel_size, flatten_factor, padding=0):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out1_channel, kernel_size=(1, kernel_size), stride=1, padding=padding),
            # nn.BatchNorm2d(out1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(out1_channel, out2_channel, kernel_size=(1, kernel_size), stride=1, padding=padding),
            # nn.BatchNorm2d(out2_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(flatten_factor * out2_channel, fc),
            nn.ReLU(inplace=True),
            nn.Linear(fc, out_classes),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out2


class NetS(nn.Module):
    def __init__(self, in_channel=9, out1_channel=32, out2_channel=64, out3_channel=128, out4_channel=256, out_classes=10, avg_factor=13, kernel_size=14):
        super(NetS, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, (1, kernel_size), stride, 0, bias=False),
                # nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, (1, kernel_size), stride, 0, groups=inp, bias=False),
                # nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(in_channel, out1_channel, (1, 2)),
            conv_dw(out1_channel, out2_channel, (1, 1)),
            conv_dw(out2_channel, out3_channel, (1, 2)),
            conv_dw(out3_channel, out4_channel, (1, 2)),
            nn.AvgPool2d((1, avg_factor)),
        )
        self.fc = nn.Linear(out4_channel, out_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_10': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'VGG11_25': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG11_25_1': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11_25_2': [16, 'M', 32, 'M', 128, 128, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11_40': [25, 'M', 51, 'M', 102, 102, 'M', 204, 204, 'M', 204, 204, 'M'],
    'VGG11_50': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11_60': [38, 'M', 76, 'M', 153, 153, 'M', 307, 307, 'M', 307, 307, 'M'],
    'VGG11_70': [44, 'M', 89, 'M', 179, 179, 'M', 358, 358, 'M', 358, 358, 'M'],
    'VGG11_s1': [63, 'M', 44, 'M', 51, 51, 'M', 86, 86, 'M', 86, 86, 'M'],
    'VGG11_bn_s1': [17, 'M', 30, 'M', 52, 52, 'M', 86, 86, 'M', 86, 86, 'M'],
    'seed': [6, 'M', 12, 'M', 25, 25, 'M', 50, 50, 'M', 50, 50, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Sequential(nn.Linear(512, 10))
        self.classifier = nn.Sequential(nn.Linear(cfg[vgg_name][-2], 10))

    def forward(self, x):
        out = self.features(x)
        out1 = out.view(out.size(0), -1)
        out = self.classifier(out1)
        return out1, out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_BN(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_BN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Sequential(nn.Linear(512, 10))
        self.classifier = nn.Sequential(nn.Linear(cfg[vgg_name][-2], 10))

    def forward(self, x):
        out = self.features(x)
        out1 = out.view(out.size(0), -1)
        out = self.classifier(out1)
        return out1, out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = VGG('VGG11_25_2')
    summary = summary(net, (3, 34, 34))

    # macs, params = get_model_complexity_info(net, (3, 34, 34), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #
    # print('abcd')



