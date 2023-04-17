import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.down1 = nn.ModuleList([
            nn.Conv2d(8, 16, 3, 2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU()    
        ])
        self.down2 = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU()    
        ])
        self.down3 = nn.ModuleList([
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()    
        ])
        self.down4 = nn.ModuleList([
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()    
        ])

        self.lin1 = nn.Conv2d(128, 64, 1)
        self.lin2 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # x: (b, 1, 80, 624) -> (b, 2, 5, 39)
        x = self.conv1(x)
        for f in self.down1:
            x = f(x)
        for f in self.down2:
            x = f(x)
        for f in self.down3:
            x = f(x)
        for f in self.down4:
            x = f(x)
        x = self.lin1(x)
        x = F.selu(x)
        x = self.lin2(x)
        return x

def gan_loss_real(x_real, device):
    loss_real = F.binary_cross_entropy_with_logits(x_real, 0.95 * torch.ones(x_real.shape).to(device))
    return loss_real

def gan_loss_fake(x_fake, device):
    loss_fake = F.binary_cross_entropy_with_logits(x_fake, torch.zeros(x_fake.shape).to(device))
    return loss_fake


def gan_loss_d(x_real, x_fake, device):
    return F.mse_loss(x_real, torch.ones(x_real.shape).to(device)) + F.mse_loss(x_fake, torch.zeros(x_fake.shape).to(device))

def gan_loss_g(x_fake, device):
    return F.mse_loss(x_fake, torch.ones(x_fake.shape).to(device))