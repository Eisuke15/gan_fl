from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm, trange

from gan import Discriminator, Generator

parser = ArgumentParser()
parser.add_argument('-e', '--nepoch', type=int, help="number of epochs to train for", default=1000)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

dataset_train = MNIST(root='data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=2)

g = Generator(args.nz, conditional=conditional).to(device)
d = Discriminator(args.nz, conditional=conditional).to(device)

g_optimizer = Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = BCEWithLogitsLoss()

fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

# pre-self training
# train local model just by using the local data
for epoch in range(args.nepoch + 1):
   
    g.train()
    d.train()

    errDs = []
    errGs = []
    D_x_seq = []
    D_G_z1_seq = []
    D_G_z2_seq = []

    for images, _ in tqdm(train_dataloader, leave=False):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        d.zero_grad()
        real_images = images.to(device)
        batch_size = real_images.size(0)
        real_label = torch.full((batch_size,), 1, dtype=real_images.dtype, device=device)
        fake_label = torch.full((batch_size,), 0, dtype=real_images.dtype, device=device)
        output = d(real_images)
        errD_real = criterion(output, real_label)
        errD_real.backward()
        D_x = torch.where(output > 0.5, 1., 0.).mean().item()
        D_x_seq.append(D_x)

        # train with fake
        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        fake = g(noise)
        output = d(fake.detach())
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        D_G_z1 = torch.where(output > 0.5, 1., 0.).mean().item()
        errD = errD_real + errD_fake
        errDs.append(errD.item())
        D_G_z1_seq.append(D_G_z1)
        d_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        g.zero_grad()
        output = d(fake)
        errG = criterion(output, real_label)
        errG.backward()
        errGs.append(errG.item())
        D_G_z2 = torch.where(output > 0.5, 1., 0.).mean().item()
        D_G_z2_seq.append(D_G_z2)
        g_optimizer.step()

    print(f'[{epoch}/{args.nepoch}] Loss_D: {np.mean(errDs):.4f} Loss_G: {np.mean(errGs):.4f} D(x): {np.mean(D_x_seq):.4f} D(G(z)): {np.mean(D_G_z1_seq):.4f} / {np.mean(D_G_z2_seq):.4f}')

    if epoch%10 == 0:
        g.eval()
        save_image(g(fixed_noise), f'images/central/e{epoch}_z{args.nz}.png')
