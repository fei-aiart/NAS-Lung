'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models.net_sphere as sp_net
import transforms as transforms
from dataloader import lunanod
import os
import argparse
import time
from models.cnn_res import *
# from utils import progress_bar
from torch.autograd import Variable
import logging
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size ')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--savemodel', type=str, default='', help='resume from checkpoint model')
parser.add_argument("--gpuids", type=str, default='all', help='use which gpu')

parser.add_argument('--num_epochs', type=int, default=700)
parser.add_argument('--num_epochs_decay', type=int, default=70)

parser.add_argument('--num_workers', type=int, default=24)

parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--lamb', type=float, default=1, help="lambda for loss2")
parser.add_argument('--fold', type=int, default=5, help="fold")

args = parser.parse_args()

CROPSIZE = 32
gbtdepth = 1
fold = args.fold
blklst = []  # ['1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-388', \
# '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-389', \
# '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957-660']
logging.basicConfig(filename='log-' + str(fold), level=logging.INFO)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
best_acc_gbt = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Cal mean std
# preprocesspath = '/media/data1/wentao/tianchi/luna16/cls/crop_v3/'
preprocesspath = '/data/xxx/LUNA/cls/crop_v3/'
# preprocesspath = '/media/jehovah/Work/data/LUNA/cls/crop_v3/'
pixvlu, npix = 0, 0
for fname in os.listdir(preprocesspath):
    # print(fname)
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname))
        pixvlu += np.sum(data)
        # print("data.shape = " + str(data.shape))
        npix += np.prod(data.shape)
pixmean = pixvlu / float(npix)
pixvlu = 0
for fname in os.listdir(preprocesspath):
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname)) - pixmean
        pixvlu += np.sum(data * data)
pixstd = np.sqrt(pixvlu / float(npix))
# pixstd /= 255
print(pixmean, pixstd)
logging.info('mean ' + str(pixmean) + ' std ' + str(pixstd))
# Datatransforms
logging.info('==> Preparing data..')  # Random Crop, Zero out, x z flip, scale,
transform_train = transforms.Compose([
    # transforms.RandomScale(range(28, 38)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []
tefnamelst = []
telabellst = []
tefeatlst = []
import pandas as pd

dataframe = pd.read_csv('./data/annotationdetclsconvfnl_v3.csv',
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]
# test id
teidlst = []
for fname in os.listdir('/data/xxx/LUNA/rowfile/subset' + str(fold) + '/'):
    # for fname in os.listdir('/media/jehovah/Work/data/LUNA/rowfile/subset' + str(fold) + '/'):

    if fname.endswith('.mhd'):
        teidlst.append(fname[:-4])
mxx = mxy = mxz = mxd = 0
for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
    mxx = max(abs(float(x)), mxx)
    mxy = max(abs(float(y)), mxy)
    mxz = max(abs(float(z)), mxz)
    mxd = max(abs(float(d)), mxd)
    if srsid in blklst: continue
    # crop raw pixel as feature
    data = np.load(os.path.join(preprocesspath, srsid + '.npy'))
    bgx = int(data.shape[0] / 2 - CROPSIZE / 2)
    bgy = int(data.shape[1] / 2 - CROPSIZE / 2)
    bgz = int(data.shape[2] / 2 - CROPSIZE / 2)
    data = np.array(data[bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])
    # feat = np.hstack((np.reshape(data, (-1,)) / 255, float(d)))
    y, x, z = np.ogrid[-CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2]
    mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
    feat = np.zeros((CROPSIZE, CROPSIZE, CROPSIZE), dtype=float)
    feat[mask] = 1
    # print(feat.shape)
    if srsid.split('-')[0] in teidlst:
        tefnamelst.append(srsid + '.npy')
        telabellst.append(int(label))
        tefeatlst.append(feat)
    else:
        trfnamelst.append(srsid + '.npy')
        trlabellst.append(int(label))
        trfeatlst.append(feat)
for idx in range(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in range(len(tefeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    tefeatlst[idx][-1] /= mxd
trainset = lunanod(preprocesspath, trfnamelst, trlabellst, trfeatlst, train=True, download=True,
                   transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20)

testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, download=True,
                  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=20)
savemodelpath = './checkpoint-' + str(fold) + '/'
train_val = np.empty(shape=0)
test_val = np.empty(shape=(0, 3))
# Model
print(args.resume)
if args.resume:

    print('==> Resuming from checkpoint..')
    print(args.savemodel)
    if args.savemodel == '':
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir(savemodelpath), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(savemodelpath + 'ckpt.t7')

    else:
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir(savemodelpath), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.savemodel)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(savemodelpath + " load success")
    print(start_epoch)
else:
    logging.info('==> Building model..')
    logging.info('args.savemodel : ' + args.savemodel)
    net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    if args.savemodel != "":
        # args.savemodel = '/home/xxx/DeepLung-master/nodcls/checkpoint-5/ckpt.t7'
        checkpoint = torch.load(args.savemodel)
        finenet = checkpoint
        Low_rankmodel_dic = net.state_dict()
        finenet = {k: v for k, v in finenet.items() if k in Low_rankmodel_dic}
        Low_rankmodel_dic.update(finenet)
        net.load_state_dict(Low_rankmodel_dic)
        print("net_loaded")

lr = args.lr


def get_lr(epoch):
    global lr
    if (epoch + 1) > (args.num_epochs - args.num_epochs_decay):
        lr -= (lr / float(args.num_epochs_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Decay learning rate to lr: {}.'.format(lr))


if use_cuda:
    net.cuda()
    if args.gpuids == 'all':
        device_ids = range(torch.cuda.device_count())
    else:
        device_ids = map(int, list(filter(str.isdigit, args.gpuids)))

    print('gpu use' + str(device_ids))
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    cudnn.benchmark = False  # True

criterion = sp_net.AngleLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# L2Loss = torch.nn.MSELoss()

# Training
def train(epoch):
    logging.info('\nEpoch: ' + str(epoch))
    net.train()
    get_lr(epoch)
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, feat) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs[0].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'

    print('ep ' + str(epoch) + ' tracc ' + str(correct.data.item() / float(total)) + ' lr ' + str(lr))
    logging.info(
        'ep ' + str(epoch) + ' tracc ' + str(correct.data.item() / float(total)) + ' lr ' + str(lr))
    np.append(train_val, correct.data.item() / float(total))


def test(epoch):
    epoch_start_time = time.time()
    global best_acc
    global best_acc_gbt
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    TP = FP = FN = TN = 0
    for batch_idx, (inputs, targets, feat) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs[0].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
        TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
        FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
        FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()

    # Save checkpoint.
    acc = 100. * correct.data.item() / total
    if acc > best_acc:
        logging.info('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(savemodelpath):
            os.mkdir(savemodelpath)
        torch.save(state, savemodelpath + 'ckpt.t7')
        best_acc = acc
    logging.info('Saving..')
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(savemodelpath):
        os.mkdir(savemodelpath)
    if epoch % 50 == 0:
        torch.save(state, savemodelpath + 'ckpt' + str(epoch) + '.t7')
    # best_acc = acc
    tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
    fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())

    print('teacc ' + str(acc) + ' bestacc ' + str(best_acc))
    print('tpr ' + str(tpr) + ' fpr ' + str(fpr))
    print('Time Taken: %d sec' % (time.time() - epoch_start_time))
    logging.info(
        'teacc ' + str(acc) + ' bestacc ' + str(best_acc))
    logging.info(
        'tpr ' + str(tpr) + ' fpr ' + str(fpr))
    np.append(test_val, [[acc, tpr, fpr]], axis=0)


if __name__ == '__main__':
    for epoch in range(start_epoch + 1, start_epoch + args.num_epochs + 1):  # 200):
        train(epoch)
        test(epoch)
    np.save(savemodelpath + "train_acc", train_val)
    np.save(savemodelpath + "test_acc", test_val)
