import json
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
parser.add_argument('-p', '--pre-nepoch', type=int, help='number of epochs of pre-self train', default=100)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=20)
parser.add_argument('--nnodes', type=int, help='number of nodes (number of labels)', default=10)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-a', '--aggregation', choices=[1, 2], type=int, help="aggregation method", default=1)
args = parser.parse_args()

n_node = args.nnodes

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])


filename=f'./contact_pattern/rwp_n10_a0500_r100_p10_s01.json'
# filename=f'./contact_pattern/cse_n10_c10_b02_tt10_tp5_s01.json'
print(f'Loading ... {filename}')
with open(filename) as f :
    contact_list=json.load(f)

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

dataset_train = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2) for subset in subsets]

generators = [Generator(args.nz).to(device) for _ in range(n_node)]
discriminators = [Discriminator(args.nz).to(device) for _ in range(n_node)]

if args.aggregation == 1:
    lr_g = 0.0001
    lr_d = 0.0004
else:
    lr_g = 0.0001 / 2
    lr_d = 0.0004 / 2
g_optimizers = [Adam(net.parameters(), lr=lr_g, betas=(0.0, 0.9)) for net in generators]
d_optimizers = [Adam(net.parameters(), lr=lr_d, betas=(0.0, 0.9)) for net in discriminators]

criterion = BCEWithLogitsLoss()

fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

# pre-self training
# train local model just by using the local data
for epoch in trange(args.pre_nepoch, desc="pre-self training epoch"):
    for node_num, (g, d, g_optimizer, d_optimizer, dataloader) in tqdm(enumerate(zip(generators, discriminators, g_optimizers, d_optimizers, train_loaders)), leave=False, total=n_node, desc="node"):
        g.train()
        d.train()
        for images, _ in tqdm(dataloader, leave=False, desc="batch"):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            d.zero_grad()
            real_cpu = images.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), 1, dtype=real_cpu.dtype, device=device)

            output = d(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = torch.where(output > 0.5, 1., 0.).mean().item()

            # train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = g(noise)
            label.fill_(0)
            output = d(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = torch.where(output > 0.5, 1., 0.).mean().item()
            errD = errD_real + errD_fake
            d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            output = d(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = torch.where(output > 0.5, 1., 0.).mean().item()
            g_optimizer.step()

        if epoch%10 == 0 or epoch == args.pre_nepoch - 1:
            g.eval()
            save_image(g(fixed_noise), f'images/wafl/pre_a{args.aggregation}_e{epoch}_z{args.nz}_n{node_num}.png')


global_generator = Generator(args.nz).to(device).state_dict()
global_discriminator = Discriminator(args.nz).to(device).state_dict()

for epoch in range(args.nepoch + 1):
    contact = contact_list[epoch]
    print(contact)

    # store updated parameters here
    updated_generators = [g.state_dict() for g in generators]
    updated_discriminators = [d.state_dict() for d in discriminators]

    # create lists of dictionary for easy reference.
    local_generators = [g.state_dict() for g in generators]
    local_discriminators = [d.state_dict() for d in discriminators]

    # exchange models
    for i, (g, d, updated_g, updated_d) in enumerate(zip(local_generators, local_discriminators, updated_generators, updated_discriminators)):
        neighbors = contact[str(i)]
        if neighbors:
            for key in g:
                updated_g[key] = sum([local_generators[neighbor][key] for neighbor in neighbors] + [g[key]])/(len(neighbors)+ 1)
            for key in d:
                updated_d[key] = sum([local_discriminators[neighbor][key] for neighbor in neighbors] + [d[key]])/(len(neighbors)+ 1)

    for node_num, (g, d, updated_g, updated_d, g_optimizer, d_optimizer, dataloader) in enumerate(zip(generators, discriminators, updated_generators, updated_discriminators, g_optimizers, d_optimizers, train_loaders)):
        
        # load updated models
        g.load_state_dict(updated_g)
        d.load_state_dict(updated_d)

        if epoch%10 == 0:
            g.eval()
            save_image(g(fixed_noise), f'images/wafl/a{args.aggregation}_e{epoch}_z{args.nz}_n{node_num}_before.png')

        g.train()
        d.train()

        # skip train when no neighbor
        if contact[str(node_num)]:
            errDs = []
            errGs = []
            D_x_seq = []
            D_G_z1_seq = []
            D_G_z2_seq = []

            for i, (images, _) in enumerate(dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                d.zero_grad()
                real_cpu = images.to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), 1, dtype=real_cpu.dtype, device=device)

                output = d(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = torch.where(output > 0.5, 1., 0.).mean().item()
                D_x_seq.append(D_x)

                # train with fake
                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                fake = g(noise)
                label.fill_(0)
                output = d(fake.detach())
                errD_fake = criterion(output, label)
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
                label.fill_(1)  # fake labels are real for generator cost
                output = d(fake)
                errG = criterion(output, label)
                errG.backward()
                errGs.append(errG.item())
                D_G_z2 = torch.where(output > 0.5, 1., 0.).mean().item()
                D_G_z2_seq.append(D_G_z2)
                g_optimizer.step()

            print(f'[{epoch}/{args.nepoch}] node: {node_num} Loss_D: {np.mean(errDs):.4f} Loss_G: {np.mean(errGs):.4f} D(x): {np.mean(D_x_seq):.4f} D(G(z)): {np.mean(D_G_z1_seq):.4f} / {np.mean(D_G_z2_seq):.4f}')

        if epoch%10 == 0:
            g.eval()
            save_image(g(fixed_noise), f'images/wafl/a{args.aggregation}_e{epoch}_z{args.nz}_n{node_num}_after.png')