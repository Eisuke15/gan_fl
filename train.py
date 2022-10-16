import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from concat import concat_image_label, concat_noise_label

criterion = BCEWithLogitsLoss()


def gen_fixed_noise(nz, device, conditional):
    gen_image_num = 70
    fixed_noise = torch.randn(gen_image_num, nz, 1, 1, device=device)
    return concat_noise_label(
        fixed_noise,
        torch.tensor([i % 10 for i in range(gen_image_num)], dtype=torch.long, device=device),
        device) if conditional else fixed_noise


def train(dataloader, g, d, g_optimizer, d_optimizer, nz, epoch, n_epoch, device, conditional=False, node=None):
    """train generator and discriminator with given dataloader for 1 epoch"""
    g.train()
    d.train()

    errDs = []
    errGs = []
    D_x_seq = []
    D_G_z1_seq = []
    D_G_z2_seq = []

    for images, labels in tqdm(dataloader, leave=False):
        
        # preparations
        real_images = images.to(device)
        labels = labels.to(device)
        batch_size = real_images.size(0)
        real_target = torch.full((batch_size,), 1, dtype=real_images.dtype, device=device)
        fake_target = torch.full((batch_size,), 0, dtype=real_images.dtype, device=device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        d.zero_grad()
        real_input = concat_image_label(real_images, labels, device) if conditional else real_images
        output = d(real_input)
        errD_real = criterion(output, real_target)
        errD_real.backward()
        D_x = torch.where(output > 0, 1., 0.).mean().item()
        D_x_seq.append(D_x)

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_label = torch.randint(10, (batch_size,), dtype=torch.long, device=device)
        fake_input_noise = concat_noise_label(noise, fake_label, device) if conditional else noise
        fake_images = g(fake_input_noise)
        fake_input = concat_image_label(fake_images, fake_label, device) if conditional else fake_images
        output = d(fake_input.detach())
        errD_fake = criterion(output, fake_target)
        errD_fake.backward()
        D_G_z1 = torch.where(output > 0, 1., 0.).mean().item()
        errD = errD_real + errD_fake
        errDs.append(errD.item())
        D_G_z1_seq.append(D_G_z1)
        d_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        g.zero_grad()
        output = d(fake_input)
        errG = criterion(output, real_target)
        errG.backward()
        errGs.append(errG.item())
        D_G_z2 = torch.where(output > 0, 1., 0.).mean().item()
        D_G_z2_seq.append(D_G_z2)
        g_optimizer.step()

    log = f'[{epoch}/{n_epoch}] Loss_D: {np.mean(errDs):.4f} Loss_G: {np.mean(errGs):.4f} D(x): {np.mean(D_x_seq):.4f} D(G(z)): {np.mean(D_G_z1_seq):.4f} / {np.mean(D_G_z2_seq):.4f}'
    if node is not None:
        log = f'node:{node:2d}: ' + log
    print(log)
