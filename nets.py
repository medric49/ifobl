import torch
from hydra.utils import to_absolute_path
from torch import nn
import torchvision

import alexnet


class EfficientNetB0Encoder(nn.Module):
    def __init__(self):
        super(EfficientNetB0Encoder, self).__init__()
        self.encoder = torchvision.models.efficientnet_b0(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        x = self.encoder.features(image)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AlexNet224(nn.Module):
    def __init__(self, hidden_dim):
        super(AlexNet224, self).__init__()

        self.alex_net = alexnet.MyAlexNetCMC()
        self.alex_net.load_state_dict(torch.load(to_absolute_path('tmp/CMC_alexnet.pth'))['model'])

        self.fc_l = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.Sigmoid()
        )

        self.fc_ab = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_l, x_ab = self.alex_net(x)
        x_l = self.fc_l(x_l)
        x_ab = self.fc_ab(x_ab)
        return x_l, x_ab


class DeconvNet84(nn.Module):
    def __init__(self, hidden_dim):
        super(DeconvNet84, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_dim * 4, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=2),  # -> 7 x 7
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),  # -> 17 x 17
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1),  # -> 40 x 40
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1)  # -> 84 x 84
        )

    def forward(self, e):
        e = e.view(e.shape[0], e.shape[1], 1, 1)
        obs = self.network(e)
        return obs


class ConvNet84(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet84, self).__init__()

        def network(in_channel, hidden_dim):
            return nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=5, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=5, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=5, stride=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(256, 512, kernel_size=7, stride=2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(512, hidden_dim * 4, kernel_size=1),
                nn.BatchNorm2d(hidden_dim * 4),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
                nn.Flatten()
            )

        self.enc_l = network(1, hidden_dim // 2)
        self.enc_ab = network(2, hidden_dim // 2)

    def forward(self, x):
        x_l, x_ab = torch.split(x, [1, 2], dim=1)
        x_l = self.enc_l(x_l)
        x_ab = self.enc_ab(x_ab)

        return x_l, x_ab