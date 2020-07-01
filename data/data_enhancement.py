import numpy as np
import pandas as pd
import os


def horizontal_flip(img):
    x, y, d = img.shape
    for i in range(y // 2):
        intermediate = img[:, i, :]
        img[:, i, :] = img[:, y - 1 - i, :]
        img[:, y - 1 - i, :] = intermediate
    return img


def vertical_flip(img):
    x, y, d = img.shape
    for i in range(x // 2):
        intermediate = img[i, :, :]
        img[i, :, :] = img[x - 1 - i, :, :]
        img[x - 1 - i, :, :] = intermediate
    return img


def deep_flip(img):
    x, y, d = img.shape
    for i in range(d // 2):
        intermediate = img[:, :, i]
        img[:, :, i] = img[:, :, d - 1 - i]
        img[:, :, d - 1 - i] = intermediate
        return img


path = "F:\\医学数据集\\LUNA\\cls\\crop_v3\\"
dataframe = pd.read_csv("F:/PycharmProjects/DeepLung/data/annotationdetclsconvfnl_v3.csv", encoding='utf-8')
data_list = dataframe.to_numpy()[1:]
enhancement_data = dataframe.to_numpy()
test_data = np.empty(shape=(0, 32, 32, 32))
train_data = np.empty(shape=(0, 32, 32, 32))
teidlst = []
for fname in os.listdir('F:\\医学数据集\\LUNA\\rowfile\\subset5' + '\\'):
    if fname.endswith('.mhd'):
        teidlst.append(fname[:-4])
for data in data_list:
    if data[0].split('-')[0] not in teidlst:
        a = np.load(path + data[0] + '.npy')
        horizontal_data = horizontal_flip(a)
        horizontal_name = path + data[0] + 'horizontal'
        horizontal_information = np.copy(data)
        horizontal_information[0] = data[0] + 'horizontal'
        enhancement_data = np.append(enhancement_data, [horizontal_information], 0)
        # np.save(horizontal_name, horizontal_data)
        vertical_data = vertical_flip(a)
        vertical_name = path + data[0] + 'vertical'
        vertical_information = np.copy(data)
        vertical_information[0] = data[0] + 'vertical'
        enhancement_data = np.append(enhancement_data, [vertical_information], 0)
        # np.save(vertical_name, vertical_data)
        deep_data = deep_flip(a)
        deep_name = path + data[0] + 'deep'
        deep_information = np.copy(data)
        deep_information[0] = data[0] + 'deep'
        enhancement_data = np.append(enhancement_data, [deep_information], 0)
        # np.save(deep_name, deep_data)
frame = pd.DataFrame(enhancement_data, index=None,
                     columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
frame.to_csv('H:\\a.csv', index=None, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
