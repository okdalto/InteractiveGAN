import torch.nn as nn
import torch
# Generator Code

'''
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.front = nn.Sequential(
            nn.Linear(100, 128, bias=True),
            nn.Linear(128, 100 * 4 * 4, bias=True),
        )

        self.main = nn.Sequential(
            nn.Conv2d(100, 512 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 1024, 4, 4
            nn.PixelShuffle(2),                                                         # b, 1024, 8, 8
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 256, 8, 8
            nn.PixelShuffle(2),                                                         # b, 256, 16, 16
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 128, 16, 16
            nn.PixelShuffle(2),                                                         # b, 128, 32, 32
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64 * 4, kernel_size=3, stride=1, padding=1, bias=True),      # b, 64, 32, 32
            nn.PixelShuffle(2),                                                         # b, 64, 64, 64
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32 * 4, kernel_size=3, stride=1, padding=1, bias=True),       # b, 32, 64, 64
            nn.PixelShuffle(2),                                                         # b, 32, 128, 128
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3, bias=True),            # b, 3, 128, 128
            nn.Sigmoid(),

        )

    def forward(self, input):        
        x = self.front(input.reshape(input.size(0), -1))
        x = x.reshape((-1, 100, 4, 4))
        x = self.main(x)
        return x
'''


# Generator Code
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz = 50
        ngf = 64
        nc = 3
        self.encoder = nn.Sequential(
            PixelNorm(),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8,        kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4,    kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2,   kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf,       kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, ngf//2,        kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d( ngf//2, nc,         kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        # x = input.reshape(input.size(0), -1)
        # x = self.encoder(x)
        # x = x.reshape((-1, 128, 1, 1))
        x = self.main(input)
        return x





'''

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz = 100
        ngf = 64
        nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(nz, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf * 1, ngf//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.Conv2d(ngf//2, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),

            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),

            nn.Conv2d(nc, nc, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nc),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        x = self.main(input)
        return x
'''
'''
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(100, 256 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 1024, 4, 4
            nn.PixelShuffle(2),                                                         # b, 1024, 8, 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 1024, 4, 4
            nn.PixelShuffle(2),                                                         # b, 1024, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 1024, 4, 4
            nn.PixelShuffle(2),                                                         # b, 1024, 8, 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 256, 8, 8
            nn.PixelShuffle(2),                                                         # b, 256, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16 * 4, kernel_size=3, stride=1, padding=1, bias=True),     # b, 128, 16, 16
            nn.PixelShuffle(2),                                                         # b, 128, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 8 * 4, kernel_size=3, stride=1, padding=1, bias=True),      # b, 64, 32, 32
            nn.PixelShuffle(2),                                                         # b, 64, 64, 64
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 4 * 4, kernel_size=3, stride=1, padding=1, bias=True),       # b, 32, 64, 64
            nn.PixelShuffle(2),                                                         # b, 32, 128, 128
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 3, kernel_size=7, stride=1, padding=3, bias=True),            # b, 3, 128, 128
            nn.Sigmoid(),

        )

    def forward(self, input):        
        # x = self.front(input.reshape(input.size(0), -1))
        # x = x.reshape((-1, 512, 4, 4))
        x = self.main(input)
        return x
'''