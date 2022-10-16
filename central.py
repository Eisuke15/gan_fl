from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from gan import Discriminator, Generator
from train import gen_fixed_noise, train

parser = ArgumentParser()
parser.add_argument('-e', '--nepoch', type=int, help="number of epochs to train for", default=1000)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-c', '--conditional', action="store_true", help="Conditional GAN")
args = parser.parse_args()

n_node = 10

conditional = args.conditional

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

dataset_train = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=2)

g = Generator(args.nz, conditional=conditional).to(device)
d = Discriminator(conditional=conditional).to(device)

g_optimizer = Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = gen_fixed_noise(args.nz, device, conditional)

# training
for epoch in range(args.nepoch + 1):
    train(dataloader, g, d, g_optimizer, d_optimizer, args.nz, epoch, args.nepoch, device, conditional)
    if epoch%10 == 0:
        g.eval()
        save_image(g(fixed_noise), f'images/central/e{epoch}_z{args.nz}.png', nrow=10)
