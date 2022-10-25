import os
import random
import shutil
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm, trange

from gan import Generator
from train import concat_noise_label

parser =  ArgumentParser()
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-s', '--num_samples', type=int, help="number of samples used for culcurating fid score", default=20000)
parser.add_argument('-c', '--conditional', action="store_true", help="Conditional GAN")
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('-e', '--epoch', type=int, help="upper limit of epoch to calculate FID", default=10000)
parser.add_argument('method', choices=['central', 'fl', 'wafl', 'self'])
args = parser.parse_args()

conditional = args.conditional
nz = args.nz
num_sumples = args.num_samples
epoch = args.epoch
method = args.method
batch_size = 256
num_batch = num_sumples // batch_size

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

G = Generator(nz, conditional=conditional).to(device)

tmpdir_name = 'tmp_' + str(datetime.now())
real_path = os.path.join(tmpdir_name, 'real')
fake_path = os.path.join(tmpdir_name, 'fake')

os.makedirs(real_path)

# prepare real images
dataset = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
indices = random.sample(list(range(60000)), batch_size * num_batch)
subset = Subset(dataset, indices=indices)
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
for i, (images, labels) in tqdm(enumerate(dataloader), total=num_batch, desc="preparing real images"):
    images = images.to(device)
    labels = labels.to(device)
    for j, (image, label) in enumerate(zip(images, labels)):
        save_image([image], os.path.join(real_path, f'{i}_{label}.png'))


epochs = list(range(0, epoch+1, 50 if method == "wafl" else 20 if method == "fl" else 10))
fid_vals = []

for e in epochs:
    g_path = f'nets/{method}/{"cgan" if conditional else "gan"}/g_e{e}_z{nz}{"_n0" if method == "wafl" else "" }.pth'
    G.load_state_dict(torch.load(g_path))

    os.makedirs(fake_path)

    # prepare fake images
    for i in trange(num_batch, desc=f"preparing fake images (epoch {e})"):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_label = torch.randint(10, (batch_size,), dtype=torch.long, device=device)
        fake_input_noise = concat_noise_label(noise, fake_label, device) if conditional else noise
        fake_images = G(fake_input_noise)

        for j, (image, label) in enumerate(zip(fake_images, fake_label)):
            save_image([image], os.path.join(fake_path, f'{batch_size * i + j}_{label}.png'))

    fid_val = calculate_fid_given_paths([real_path, fake_path], 256, device, 2048, 8)
    fid_vals.append(fid_val)
    shutil.rmtree(fake_path)

shutil.rmtree(tmpdir_name)

basefilename = f'{method}_{"cgan" if conditional else "gan"}_e{epoch}_z{nz}'
np.save(os.path.join("graphs", basefilename + ".npy"), np.array([epochs, fid_vals]))

plt.plot(epochs, fid_vals)
plt.ylabel('FID score')
plt.xlabel('Epoch')
plt.savefig(os.path.join("graphs", basefilename + ".png"), bbox_inches='tight')
