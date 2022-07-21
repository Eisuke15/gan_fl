from torch import nn

class Generator(nn.Module):

    def __init__(self, z_dim=20):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, kernel_size=4, stride=1),  # 128, 4, 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),  # 64, 7, 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))


        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 1, 28, 28
            nn.Tanh())
        # 注意：白黒画像なので出力チャネルは1つだけ

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64, 14, 14
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #128, 7, 7
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 256, 4, 4
            nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4, stride=1), # 1, 1, 1
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last(out)
        return out.view(-1)


if __name__ == "__main__":
    from torchinfo import summary
    summary(Generator(), (256, 20, 1, 1))
    summary(Discriminator(), (256, 1, 28, 28))
