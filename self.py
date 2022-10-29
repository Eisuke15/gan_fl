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
parser.add_argument('-e', '--nepoch', type=int, help="number of epochs to train for", default=1000)
parser.add_argument('-z', '--nz', type=int, help='size of the latent z vector', default=16)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
parser.add_argument('-c', '--conditional', action="store_true", help="Conditional GAN")
args = parser.parse_args()

nz = args.nz
conditional = args.conditional
nepoch = args.nepoch
n_node = 10

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
for epoch in range(nepoch + 1):
    for node, (g, d, g_optimizer, d_optimizer, dataloader) in enumerate(zip(generators, discriminators, g_optimizers, d_optimizers, train_loaders)):
        train(dataloader, g, d, g_optimizer, d_optimizer, nz, epoch, nepoch, device, conditional, node)
        if epoch%10 == 0:
            g.eval()
            save_image(g(fixed_noise), f'images/self/{"cgan" if conditional else "gan"}/z{args.nz}_n{node}_e{epoch}.png', nrow=10)
            torch.save(g.state_dict(), f'nets/self/{"cgan" if conditional else "gan"}/g_e{epoch}_z{nz}.pth')
