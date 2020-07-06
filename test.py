import torch
import numpy as np
import os
import transforms as transforms
import pandas as pd
from dataloader import lunanod
from torch.autograd import Variable
from itertools import combinations, permutations
import logging
import pandas as pd
import argparse


def load_data(test_data_path, preprocess_path, fold, batch_size, num_workers):
    test_data_path = '/data/xxx/LUNA/rowfile/subset'
    crop_size = 32
    black_list = []

    preprocess_path = '/data/xxx/LUNA/cls/crop_v3'
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
    for idx in range(len(test_feat_list)):
        test_feat_list[idx][-1] /= mxd

    test_set = lunanod(preprocess_path, test_file_name_list, test_label_list, test_feat_list, train=False,
                       download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def load_module(path):
    checkpoint = torch.load(path)
    net = checkpoint['net']
    net.cuda()
    return net


def get_targets(test_loader):
    target_list = np.empty(shape=0)
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        target_list = np.append(target_list, targets)
    target_list = target_list.astype(int)
    return target_list


def get_permutations(model_list, count, top_count):
    result = []
    for i in permutations(model_list, count):
        result.append(list(i))
        if result.__len__() >= top_count:
            return result
    return result


def test_module(module, test_loader):
    module.eval()
    total = 0
    correct = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = module(inputs)
        prediction = 0
        if not isinstance(outputs, tuple):
            _, prediction = torch.max(outputs.data, 1)
            # print(prediction.shape)
            # print('1')
        else:
            _, prediction = torch.max(outputs[0].data, 1)
            # print('2')
        TP += ((prediction == 1) & (targets.data == 1)).cpu().sum()
        TN += ((prediction == 0) & (targets.data == 0)).cpu().sum()
        FN += ((prediction == 0) & (targets.data == 1)).cpu().sum()
        FP += ((prediction == 1) & (targets.data == 0)).cpu().sum()
    tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
    fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())
    acc = (TP.data.item()+TN.data.item()) / (TP.data.item()+TN.data.item()+FN.data.item()+FP.data.item())
    print(f'acc:{acc}')
    print('tpr ' + str(tpr) + ' fpr ' + str(fpr))


def get_predicted(result_array):
    positive_array = result_array == 1
    negative_array = result_array == 0
    positive_count = np.sum(positive_array, axis=0)
    negative_count = np.sum(negative_array, axis=0)
    predicted = positive_count > negative_count
    return predicted.astype(int)


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--model_path', type=str,
                    default='/data/fuhao/PartialOrderPrunning/[4,4,[4, 8, 16, 16, 16], [32, 128], [128]]/checkpoint-5/ckpt.t7',
                    help='ckpt.t7')
parser.add_argument('--fold', type=int, default=5, help='1-5')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--test_data_path', type=str, default='/data/xxx/LUNA/rowfile/subset')
parser.add_argument('--preprocess_path', type=str, default='/data/xxx/LUNA/cls/crop_v3')
args = parser.parse_args()

if __name__ == '__main__':
    fold = args.fold
    batch_size = args.batch_size
    num_workers = args.num_workers
    test_data_path = args.test_data_path
    preprocess_path = args.preprocess_path
    net = load_module(args.model_path)
    test_data_loader = load_data(test_data_path, preprocess_path, fold, batch_size, num_workers)
    test_module(net, test_data_loader)
