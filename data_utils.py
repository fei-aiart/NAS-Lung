from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
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
import copy


def load_data(trained_data_path, test_data_path, fold, batch_size, num_workers):
    crop_size = 32
    black_list = []

    preprocess_path = trained_data_path
    pix_value, npix = 0, 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list:
                continue
            data = np.load(os.path.join(preprocess_path, file_name))
            pix_value += np.sum(data)
            npix += np.prod(data.shape)
    pix_mean = pix_value / float(npix)
    pix_value = 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list: continue
            data = np.load(os.path.join(preprocess_path, file_name)) - pix_mean
            pix_value += np.sum(data * data)
    pix_std = np.sqrt(pix_value / float(npix))
    print(pix_mean, pix_std)
    transform_train = transforms.Compose([
        # transforms.RandomScale(range(28, 38)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
        transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),  # need to cal mean and std, revise norm func
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),
    ])

    # load data list
    train_file_name_list = []
    train_label_list = []
    train_feat_list = []
    test_file_name_list = []
    test_label_list = []
    test_feat_list = []

    data_frame = pd.read_csv('./data/annotationdetclsconvfnl_v3.csv',
                             names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

    all_list = data_frame['seriesuid'].tolist()[1:]
    label_list = data_frame['malignant'].tolist()[1:]
    crdx_list = data_frame['coordX'].tolist()[1:]
    crdy_list = data_frame['coordY'].tolist()[1:]
    crdz_list = data_frame['coordZ'].tolist()[1:]
    dim_list = data_frame['diameter_mm'].tolist()[1:]
    # test id
    test_id_list = []
    for file_name in os.listdir(test_data_path + str(fold) + '/'):

        if file_name.endswith('.mhd'):
            test_id_list.append(file_name[:-4])
    mxx = mxy = mxz = mxd = 0
    for srsid, label, x, y, z, d in zip(all_list, label_list, crdx_list, crdy_list, crdz_list, dim_list):
        mxx = max(abs(float(x)), mxx)
        mxy = max(abs(float(y)), mxy)
        mxz = max(abs(float(z)), mxz)
        mxd = max(abs(float(d)), mxd)
        if srsid in black_list:
            continue
        # crop raw pixel as feature
        data = np.load(os.path.join(preprocess_path, srsid + '.npy'))
        bgx = int(data.shape[0] / 2 - crop_size / 2)
        bgy = int(data.shape[1] / 2 - crop_size / 2)
        bgz = int(data.shape[2] / 2 - crop_size / 2)
        data = np.array(data[bgx:bgx + crop_size, bgy:bgy + crop_size, bgz:bgz + crop_size])
        y, x, z = np.ogrid[-crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2]
        mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
        feat = np.zeros((crop_size, crop_size, crop_size), dtype=float)
        feat[mask] = 1
        if srsid.split('-')[0] in test_id_list:
            test_file_name_list.append(srsid + '.npy')
            test_label_list.append(int(label))
            test_feat_list.append(feat)
        else:
            train_file_name_list.append(srsid + '.npy')
            train_label_list.append(int(label))
            train_feat_list.append(feat)
    for idx in range(len(train_feat_list)):
        train_feat_list[idx][-1] /= mxd
    for idx in range(len(test_feat_list)):
        test_feat_list[idx][-1] /= mxd
    train_set = lunanod(preprocess_path, train_file_name_list, train_label_list, train_feat_list, train=True,
                        download=True,
                        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = lunanod(preprocess_path, test_file_name_list, test_label_list, test_feat_list, train=False,
                       download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_module(net, use_cuda, train_loader, optimizer, criterion, log, lr, config, epoch):
    net.train()

    for i in range(epoch):
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, feat) in enumerate(train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        print(
            'ep ' + str(i) + str(config) + ' tracc ' + str(correct.data.item() / float(total)) + ' lr ' + str(lr))

        log.info(
            'ep ' + str(i) + str(config) + ' tracc ' + str(correct.data.item() / float(total)) + ' lr ' + str(lr))
    return net


def my_test_module(net, use_cuda, test_loader, criterion, log):
    epoch_start_time = time.time()
    # global best_acc
    # global best_acc_gbt
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    TP = FP = FN = TN = 0
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
        TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
        FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
        FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()

    acc = 100. * correct.data.item() / total

    tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
    fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())

    print('teacc ' + str(acc))
    print('tpr ' + str(tpr) + ' fpr ' + str(fpr))
    print('Time Taken: %d sec' % (time.time() - epoch_start_time))
    log.info(
        'teacc ' + str(acc))
    log.info(
        'tpr ' + str(tpr) + ' fpr ' + str(fpr))
    return acc


def get_acc(net, use_cuda, train_loader, test_loader, optimizer, criterion, log, lr, config, epoch):
    net = train_module(net, use_cuda, train_loader, optimizer, criterion, log, lr, config, epoch)
    acc = my_test_module(net, use_cuda, test_loader, criterion, log)
    return acc


def net_to_cuda(net, use_gpu, gpu_ids):
    if use_gpu:
        net.cuda()
        if gpu_ids == 'all':
            device_ids = range(torch.cuda.device_count())
        else:
            device_ids = list(map(int, list(filter(str.isdigit, gpu_ids))))

        print('gpu use' + str(device_ids))
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    return net


def get_module_lat(module, input_shape):
    x = torch.randn(input_shape)
    start = time.time()
    y = module(x)
    print(y)
    end = time.time()
    return end - start


def get_yw(modules, module):
    module_acc = module[1]
    tmp_modules = copy.deepcopy(modules)
    # original_module_index = np.where(tmp_modules[:, 0] == [module[0]])[0]
    # tmp_modules = np.delete(tmp_modules, original_module_index, 0)
    tmp_modules = tmp_modules.tolist()
    tmp_modules.remove(module.tolist())
    tmp_modules = np.array(tmp_modules)
    if tmp_modules.size > 0:
        modules_acc = tmp_modules[:, 1]
        tmp = np.where(modules_acc >= module_acc)[0]
        better_modules = tmp_modules[tmp]
        if better_modules.size > 0:
            min_lat_index = np.argmin(better_modules[:, 2])
            return better_modules[min_lat_index].tolist()
    return []


# a = np.array([[[[1, 2, 3], [12, 3], [1, 2]], 0.2, 0.3], [[[1, 2, 3], [12, 3], [1, 2]], 0.3, 0.3]])
# # a = np.array([[[[1, 2, 3], [12, 3], [1, 2]], 0.2, 0.3]])
# get_yw(a, a[1])
# # for i in a:
# #     print(i)
