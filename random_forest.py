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


def load_data(fold, batch_size, num_workers):
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


def load_module(module_config, set_num):
    path = f'/data/fuhao/PartialOrderPrunning/{module_config}/checkpoint-{set_num}/ckpt.t7'
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


def test_module(module_config, set_num, test_loader):
    module = load_module(module_config, set_num)
    module.eval()
    result = np.empty(shape=0)
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = module(inputs)
        if not isinstance(outputs, tuple):
            _, predicted = torch.max(outputs.data, 1)
        else:
            _, predicted = torch.max(outputs[0].data, 1)
        result = np.append(result, predicted)
    return result


def get_predicted(result_array):
    positive_array = result_array == 1
    negative_array = result_array == 0
    positive_count = np.sum(positive_array, axis=0)
    negative_count = np.sum(negative_array, axis=0)
    predicted = positive_count > negative_count
    return predicted.astype(int)


if __name__ == '__main__':
    run_result = np.empty(shape=(0, 20))
    top_count = 20
    module_list = np.load('data/model.npy')
    module_list = list(filter(lambda x: '[32,64,[' in x, module_list))
    logging.basicConfig(filename='modelfusion_huge_log', level=logging.INFO)
    save_excel = 'modelfusion_huge'
    for i in range(3, 20):
        if i % 2 == 1:
            permutations_result = get_permutations(module_list[:i + 4], i, top_count)
            num = 0
            for modules in permutations_result:
                num += 1
                logging.info(f'model_count={i}')
                print(f'model_count={i}')
                logging.info(f'num:{num}')
                print(f'num:{num}')
                logging.info(modules)
                print(modules)
                line = []
                for fold in range(6):
                    test_loader = load_data(fold, 8, 20)
                    targets = get_targets(test_loader)
                    length = targets.shape[0]
                    all_result = np.empty(shape=(0, length))
                    for module_config in modules:
                        result = test_module(module_config, fold, test_loader)
                        all_result = np.append(all_result, [result], axis=0)
                    predicted = get_predicted(all_result)
                    TP = np.sum((predicted == 1) & (targets == 1))
                    TN = np.sum((predicted == 0) & (targets == 0))
                    FN = np.sum((predicted == 0) & (targets == 1))
                    FP = np.sum((predicted == 1) & (targets == 0))
                    tpr = 100. * TP / (TP + FN)
                    fpr = 100. * FP / (FP + TN)
                    acc = 100. * np.sum(predicted == targets) / length
                    line.append(acc)
                    line.append(tpr)
                    line.append(fpr)
                    logging.info(f'set={fold}')
                    print(f'set={fold}')
                    logging.info(f'acc={acc}')
                    print(f'acc={acc}')
                    logging.info(f'tpr={tpr} fpr={fpr}')
                    print(f'tpr={tpr} fpr={fpr}')
                run_result = np.append(run_result, np.array(line))
    np.save('run_result_huge', run_result)
    df = pd.DataFrame(data=run_result,
                      columns=['module_count', 'module_config',
                               'fold-0-acc', 'fold-0-tpr', 'fold-0-fpr',
                               'fold-1-acc', 'fold-1-tpr', 'fold-1-fpr',
                               'fold-2-acc', 'fold-2-tpr', 'fold-2-fpr',
                               'fold-3-acc', 'fold-3-tpr', 'fold-3-fpr',
                               'fold-4-acc', 'fold-4-tpr', 'fold-4-fpr',
                               'fold-5-acc', 'fold-5-tpr', 'fold-5-fpr'],
                      index=None)
    df.to_excel(save_excel)
