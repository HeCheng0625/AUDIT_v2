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
        # self.mid1 = nn.ModuleList([
        #     nn.Conv2d(16, 16, 3, 1, padding=1),
        #     nn.GroupNorm(8, 16),
        #     nn.SiLU()            
        # ])
        self.down2 = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU()    
        ])
        # self.mid2 = nn.ModuleList([
        #     nn.Conv2d(32, 32, 3, 1, padding=1),
        #     nn.GroupNorm(8, 32),
        #     nn.SiLU()    
        # ])
        self.down3 = nn.ModuleList([
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()    
        ])
        # self.mid3 = nn.ModuleList([
        #     nn.Conv2d(64, 64, 3, 1, padding=1),
        #     nn.GroupNorm(8, 64),
        #     nn.SiLU()    
        # ])
        self.down4 = nn.ModuleList([
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()    
        ])
        # self.mid4 = nn.ModuleList([
        #     nn.Conv2d(128, 128, 3, 1, padding=1),
        #     nn.GroupNorm(8, 128),
        #     nn.SiLU()    
        # ])
        self.lin1 = nn.Conv2d(128, 64, 1)
        self.lin2 = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # x: (b, 1, 80, 624) -> (b, 2, 5, 39)
        x = self.conv1(x)
        for f in self.down1:
            x = f(x)
        # res = x
        # for f in self.mid1:
        #     x = f(x)
        # x = res + x
        for f in self.down2:
            x = f(x)
        # res = x
        # for f in self.mid2:
        #     x = f(x)
        # x = res + x
        for f in self.down3:
            x = f(x)
        # res = x
        # for f in self.mid3:
        #     x = f(x)
        # x = res + x
        for f in self.down4:
            x = f(x)
        # res = x
        # for f in self.mid4:
        #     x = f(x)
        # x = res + x
        x = self.lin1(x)
        x = F.selu(x)
        x = self.lin2(x)
        return x

def gan_loss_real(x_real, device):
    loss_real = F.cross_entropy(x_real, torch.ones((x_real.shape[0], x_real.shape[-2], x_real.shape[-1])).to(torch.int64).to(device))
    return loss_real

def gan_loss_fake(x_fake, device):
    loss_fake = F.cross_entropy(x_fake, torch.zeros((x_fake.shape[0], x_fake.shape[-2], x_fake.shape[-1])).to(torch.int64).to(device))
    return loss_fake