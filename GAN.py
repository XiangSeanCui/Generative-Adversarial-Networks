#
#This is the code for training a discriminator with the generator
#



import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import time
import argparse
import datetime
import logging

parser = argparse.ArgumentParser(description='GAN CIFAR-10 Training')
parser.add_argument('--resume', '-r', default = "./checkpoint.pth.tar", help='Resume from checkpoint')
parser.add_argument('--num_epochs', '-n', default=300, type=int, help='total epoch')
args = parser.parse_args()

batch_size_train = 128
batch_size_test = 100
n_classes = 10

transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=0.1*torch.randn(1),
                contrast=0.1*torch.randn(1),
                saturation=0.1*torch.randn(1),
                hue=0.1*torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='../data',
                                             train=True,
                                             transform=transform_train,
                                             download=False)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='../data',
                                             train=False,
                                             transform=transform_test,
                                             download=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test,shuffle=False)

print("Model Setup")

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        #conv1
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln1 = nn.LayerNorm([128, 32, 32], elementwise_affine = False)
        self.leaky1 = nn.LeakyReLU()
        #conv2
        self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.ln2 = nn.LayerNorm([128, 16, 16], elementwise_affine = False)
        self.leaky2 = nn.LeakyReLU()
        #conv3
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln3 = nn.LayerNorm([128, 16, 16], elementwise_affine = False)
        self.leaky3 = nn.LeakyReLU()
        #conv4
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.ln4 = nn.LayerNorm([128, 8, 8], elementwise_affine = False)
        self.leaky4 = nn.LeakyReLU()
        #conv5
        self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln5 = nn.LayerNorm([128, 8, 8], elementwise_affine = False)
        self.leaky5 = nn.LeakyReLU()
        #conv6
        self.conv6 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln6 = nn.LayerNorm([128, 8, 8], elementwise_affine = False)
        self.leaky6 = nn.LeakyReLU()
        #conv7
        self.conv7 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln7 = nn.LayerNorm([128, 8, 8], elementwise_affine = False)
        self.leaky7 = nn.LeakyReLU()
        #conv8
        self.conv8 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.ln8 = nn.LayerNorm([128, 4, 4], elementwise_affine = False)
        self.leaky8 = nn.LeakyReLU()
        #fc
        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.leaky1(self.ln1(self.conv1(x)))
        x = self.leaky2(self.ln2(self.conv2(x)))
        x = self.leaky3(self.ln3(self.conv3(x)))
        x = self.leaky4(self.ln4(self.conv4(x)))
        x = self.leaky5(self.ln5(self.conv5(x)))
        x = self.leaky6(self.ln6(self.conv6(x)))
        x = self.leaky7(self.ln7(self.conv7(x)))
        x = self.leaky8(self.ln8(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size = 4)
        x = x.view(x.size(0), -1)
        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return (output1, output2)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        #fc1
        self.fc1 = nn.Linear(100, 128*4*4)
        #conv1
        self.conv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        #conv2
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        #conv3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        #conv4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        #conv5
        self.conv5 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        #conv6
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        #conv7
        self.conv7 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.relu7 = nn.ReLU()
        #conv8
        self.conv8 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 128, 4, 4)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        x = torch.tanh(x)
        return x

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size_train, 1)
    alpha = alpha.expand(batch_size_train, int(real_data.nelement()/batch_size_train)).contiguous()
    alpha = alpha.view(batch_size_train, 3, DIM, DIM)
    if use_cuda:
        alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size_train, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

use_cuda = torch.cuda.is_available()
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,100))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise)
if use_cuda:
    save_noise = save_noise.cuda()

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

aD = discriminator()
aG = generator()
use_cuda = torch.cuda.is_available()

if use_cuda:
    aD.cuda()
    aG.cuda()
optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))
criterion = nn.CrossEntropyLoss()

start_epoch = 0
train_accu_epoch = []
test_accu_epoch = []
learning_rate = 0.0001

if args.resume:
    print("Resume from checkpoint")
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        aD.load_state_dict(checkpoint['aD_state_dict'])
        aG.load_state_dict(checkpoint['aG_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        train_accu_epoch = checkpoint['train_accu_epoch']
        test_accu_epoch = checkpoint['test_accu_epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, (checkpoint['epoch'] + 1)))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



print("Training Begins")
start_time = time.time()

# Train the model
for epoch in range(start_epoch, args.num_epochs):
    aG.train()
    aD.train()
    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0] < batch_size_train):
            continue
        # train G
        for p in aD.parameters():
            p.requires_grad_(False)
        aG.zero_grad()
        label = np.random.randint(0,n_classes,batch_size_train)
        noise = np.random.normal(0,1,(batch_size_train,100))
        label_onehot = np.zeros((batch_size_train,n_classes))
        label_onehot[np.arange(batch_size_train), label] = 1
        noise[np.arange(batch_size_train), :n_classes] = label_onehot[np.arange(batch_size_train)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise)
        fake_label = Variable(torch.from_numpy(label))
        if use_cuda:
            noise, fake_label = noise.cuda(), fake_label.cuda()
        fake_data = aG(noise)
        gen_source, gen_class  = aD(fake_data)
        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, fake_label)
        gen_cost = -gen_source + gen_class
        gen_cost.backward()
        if(epoch>6):
            for group in optimizer_g.param_groups:
                for p in group['params']:
                    state = optimizer_g.state[p]
                    if('step' in state and state['step']>1024):
                        state['step']=1000
        optimizer_g.step()
        # train D
        for p in aD.parameters():
            p.requires_grad_(True)
        aD.zero_grad()
        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size_train)
        noise = np.random.normal(0,1,(batch_size_train,100))
        label_onehot = np.zeros((batch_size_train,n_classes))
        label_onehot[np.arange(batch_size_train), label] = 1
        noise[np.arange(batch_size_train), :n_classes] = label_onehot[np.arange(batch_size_train)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise, fake_label = Variable(noise), Variable(torch.from_numpy(label))
        if use_cuda:
            noise, fake_label = noise.cuda(), fake_label.cuda()
        with torch.no_grad():
            fake_data = aG(noise)
        disc_fake_source, disc_fake_class = aD(fake_data)
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)
        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch)
        real_label = Variable(Y_train_batch)
        if use_cuda:
            real_data = real_data.cuda()
            real_label = real_label.cuda()
        disc_real_source, disc_real_class = aD(real_data)
        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size_train))*100.0
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)
        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()
        if(epoch>6):
            for group in optimizer_d.param_groups:
                for p in group['params']:
                    state = optimizer_d.state[p]
                    if('step' in state and state['step']>1024):
                        state['step']=1000
        optimizer_d.step()
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx%50)==0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1),
                                    "%.2f" % np.mean(loss2),
                                    "%.2f" % np.mean(loss3),
                                    "%.2f" % np.mean(loss4),
                                    "%.2f" % np.mean(loss5),
                                    "%.2f" % np.mean(acc1))
    train_accu_epoch.append(np.mean(acc1))
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch), Variable(Y_test_batch)
            if use_cuda:
                X_test_batch, Y_test_batch = X_test_batch.cuda(), Y_test_batch.cuda()
            with torch.no_grad():
                _, output = aD(X_test_batch)
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size_test))*100.0
            test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)
        test_accu_epoch.append(np.mean(test_accu))
    print("Testing",accuracy_test, time.time()-start_time)
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()
    fig = plot(samples)
    plt.savefig('../outputnew/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)
    state = {
            'epoch': epoch,
            'aD_state_dict': aD.state_dict(),
            'aG_state_dict': aG.state_dict(),
            'train_accu_epoch': train_accu_epoch,
            'test_accu_epoch': test_accu_epoch,
            'optimizer_g' : optimizer_g.state_dict(),
            'optimizer_d' : optimizer_d.state_dict()}
    torch.save(state, 'checkpoint.pth.tar')

torch.save(aD,'discriminator.model')
torch.save(aG,'generator.model')

a = np.array(train_accu_epoch)
b = np.array(test_accu_epoch)
df = pd.DataFrame({"train_accu" : a, "test_accu" : b})
df.to_csv("result.csv", index=False)
