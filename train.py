from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import CustomDataset
from models import Model
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    try:
        torch.cuda.device(args.cuda_device)
    except:
        print('Using default CUDA device..')
        pass

writer = SummaryWriter(log_dir=args.log_dir)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = CustomDataset()
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(args.train_test_split * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices, val_indices \
    = indices[2*split:], indices[:split], indices[split:2*split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, 
                          batch_size=args.batch_size,
                          sampler=train_sampler)
test_loader = DataLoader(dataset, 
                         batch_size=args.batch_size,
                         sampler=test_sampler)
val_loader = DataLoader(dataset, 
                        batch_size=args.batch_size,
                        sampler=val_sampler)

model = Model()
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    features, labels = next(iter(train_loader))
    if args.cuda:
        features = features.cuda()
        labels = labels.cuda()

    output = model(features, labels)
    loss_train = F.nll_loss(output, label)
    acc_train = accuracy(output, label)

    writer.add_scalar('Training Loss', loss_train.data.item(), epoch)
    writer.add_scalar('Training Accuracy', acc_train.data.item(), epoch)

    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train.data.item()


def evaluate():
    model.eval()

    features, labels = next(iter(val_loader))
    if args.cuda:
        features = features.cuda()
        labels = labels.cuda()

    output = model(features, labels)
    loss_val = F.nll_loss(output, label)
    acc_val = accuracy(output, label)

    writer.add_scalar('Training Loss', loss_val.data.item(), epoch)
    writer.add_scalar('Training Accuracy', acc_val.data.item(), epoch)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()))


def compute_test():
    model.eval()

    features, labels = next(iter(test_loader))
    if args.cuda:
        features = features.cuda()
        labels = labels.cuda()

    output = model(features, labels)
    loss_val = F.nll_loss(output, label)
    acc_val = accuracy(output, label)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_test: {:.4f}'.format(loss_test.data.item()),
          'acc_test: {:.4f}'.format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    if epoch % 20 == 0:
        evaluate()
    torch.save(model.state_dict(), '{}/{}.pkl'.format(args.model_dir, epoch))

    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('{}/*.pkl'.format(args.model_dir))
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('{}/*.pkl'.format(args.model_dir))
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(
    torch.load('{}/{}.pkl'.format(args.model_dir, best_epoch)))

# Testing
compute_test()
writer.close()