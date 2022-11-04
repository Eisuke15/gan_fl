from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from gan import Discriminator, Generator
from train import gen_fixed_noise, train

parser = ArgumentParser()
parser.add_argument('-e', '--nepoch', type=int, help="number of epochs to train for", default=10000)
parser.add_argument('-p', '--pre-nepoch', type=int, help='number of epochs of pre-self train', default=100)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-c', '--conditional', action="store_true", help="Conditional GAN")
args = parser.parse_args()

n_node = 10
nz = args.nz
conditional = args.conditional

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

indices=torch.load('./noniid_filter/filter_r90_s01.pt')

dataset_train = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2) for subset in subsets]

generators = [Generator(nz, conditional=conditional).to(device) for _ in range(n_node)]
discriminators = [Discriminator(conditional=conditional).to(device) for _ in range(n_node)]

lr = 0.0002
g_optimizers = [Adam(net.parameters(), lr=lr, betas=(0.5, 0.999)) for net in generators]
d_optimizers = [Adam(net.parameters(), lr=lr, betas=(0.5, 0.999)) for net in discriminators]

fixed_noise = gen_fixed_noise(nz, device, conditional)

# pre-self training
# train local model just by using the local data
for epoch in range(args.pre_nepoch):
    for node, (g, d, g_optimizer, d_optimizer, dataloader) in enumerate(zip(generators, discriminators, g_optimizers, d_optimizers, train_loaders)):
        train(dataloader, g, d, g_optimizer, d_optimizer, nz, epoch, args.pre_nepoch, device, conditional, node)
        if epoch%10 == 0 or epoch == args.pre_nepoch - 1:
            g.eval()
            save_image(g(fixed_noise), f'images/fl/{"cgan" if conditional else "gan"}/ep{epoch}_z{args.nz}_n{node}.png', nrow=10)


global_generator = Generator(nz, conditional=conditional).to(device).state_dict()
global_discriminator = Discriminator(conditional=conditional).to(device).state_dict()

for epoch in range(args.nepoch + 1):
    new_global_generator = global_generator.copy()
    new_global_discriminator = global_discriminator.copy()

    # aggregate models
    for g, d in zip(generators, discriminators):
        g_parameters = g.state_dict()
        d_parameters = d.state_dict()
        for key in new_global_generator:
            new_global_generator[key] = new_global_generator[key] + (g_parameters[key] - global_generator[key]) / n_node
        for key in new_global_discriminator:
            new_global_discriminator[key] = new_global_discriminator[key] + (d_parameters[key] - global_discriminator[key]) / n_node

    global_generator = new_global_generator
    global_discriminator = new_global_discriminator

    for node, (g, d, g_optimizer, d_optimizer, dataloader) in enumerate(zip(generators, discriminators, g_optimizers, d_optimizers, train_loaders)):

        g.load_state_dict(global_generator)
        d.load_state_dict(global_discriminator)

        train(dataloader, g, d, g_optimizer, d_optimizer, nz, epoch, args.nepoch, device, conditional, node)

    if epoch%50 == 0:
        gen = Generator(nz, conditional=conditional).to(device)
        gen.load_state_dict(global_generator)
        gen.eval()
        save_image(gen(fixed_noise), f'images/fl/{"cgan" if conditional else "gan"}/e{epoch}_z{nz}_global.png', nrow=10)
        torch.save(g.state_dict(), f'nets/fl/{"cgan" if conditional else "gan"}/g_e{epoch}_z{nz}.pth')
