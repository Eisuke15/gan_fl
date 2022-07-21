import os
import random
from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument('--bs', type=int, help="calculation batch size", default=32)
parser.add_argument('--seed', type=int, help='random seed', default=1)
parser.add_argument('--ratio', type=int, help='noniid ratio (%)', default=90)
parser.add_argument('--nnodes', type=int, help='number of nodes (number of labels)', default=10)
args = parser.parse_args()

random.seed(args.seed)

dir = './noniid_filter'
filename=os.path.join(dir, f'filter_r{args.ratio:02d}_s{args.seed:02d}.pt')
print(f'Generating NonIID filter ... {filename}')

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())

trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=2)

indices = [[] for i in range (args.nnodes)]

index = 0
for data in trainloader :
    _, y = data
    y = y.tolist()

    for i, label in enumerate(y):
        global_index = index + i

        # 指定した確率で素直にlabel番目のノードに割り振る。
        if random.randint(0, 99) < args.ratio:
            indices[label].append(global_index)

        # そうでない場合はランダムにlabel番目以外のノードに割り振る。
        else:
            # 0 ~ label-1, label+1 ~ nnodeまでの整数の内からランダムに一つ選ぶ
            n = random.choice([j for j in range(0, args.nnodes) if not j == label])
            indices[n].append(global_index)

    index += len(y)

torch.save(indices,filename)
print('Done')
