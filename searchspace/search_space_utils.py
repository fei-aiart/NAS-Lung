import numpy as np
import copy
import time
import torch
import copy
import itertools


def get_all_search_space(min_len, max_len, channel_range):
    all_search_space = []
    max_array = get_search_space(max_len, channel_range)
    max_array = np.array(max_array)
    for i in range(min_len, max_len+1):
        new_array = max_array[:, :i]
        repeat_list = new_array.tolist()
        # remove repeated list from lists
        new_list = remove_repeated_element(repeat_list)
        for list in new_list:
            for first_split in range(1, i -1):
                for second_split in range(first_split + 1, i):
                    all_search_space.append(
                        [list[:first_split], list[first_split:second_split], list[second_split:]])
    return all_search_space


def get_limited_search_space(min_len, max_len, channel_range):
    max_len = max_len+1
    all_search_space = []
    max_array = get_search_space(max_len, channel_range)
    max_array = np.array(max_array)
    for i in range(min_len, max_len):
        new_array = max_array[:, :i]
        repeat_list = new_array.tolist()
        # remove repeated list from lists
        new_list = remove_repeated_element(repeat_list)
        for list in new_list:
            for first_split in range(i // 4, i - i // 2 + 1):
                for second_split in range(first_split + i // 4, i - i // 4 + 1):
                    all_search_space.append(
                        [list[:first_split], list[first_split:second_split], list[second_split:]])
    return all_search_space


def get_search_space(max_len, channel_range, search_space=[], now=0):
    result = []
    if now == 0:
        for i in channel_range:
            result.append([i])
    else:
        for i in search_space:
            larger_channel = get_larger_channel(channel_range, i[-1])
            for m in larger_channel:
                tmp = i.copy()
                tmp.append(m)
                result.append(tmp)
    now = now + 1
    if now < max_len:
        return get_search_space(max_len, channel_range, search_space=result, now=now)
    else:
        return result


def get_larger_channel(channel_range, channel_num):
    result = filter(lambda x: x >= channel_num, channel_range)
    return list(result)


def get_smaller_channel(channel, channel_range):
    return list(filter(lambda x: x < channel, channel_range))


def get_shallower_module(min_len, module_config, shallower_module=[]):
    new_module_config = []
    for config in module_config:
        for m in range(len(config)):
            if type(config[m]) is not int:
                if len(config[m]) > 1:
                    for n in range(len(config[m])):
                        tmp = copy.deepcopy(config)
                        del tmp[m][n]
                        new_module_config.append(tmp)
    new_module_config = remove_repeated_element(new_module_config)
    shallower_module.extend(new_module_config)
    # sum(len(x) for x in a):get the count of all element
    if shallower_module != []:
        count = sum(len(x) for x in shallower_module[-1])
        if count > min_len:
            return get_shallower_module(min_len, new_module_config, shallower_module)
        else:
            return shallower_module
    else:
        return []


def remove_repeated_element(repeated_list):
    repeated_list.sort()
    new_list = [repeated_list[k] for k in range(len(repeated_list)) if
                k == 0 or repeated_list[k] != repeated_list[k - 1]]
    return new_list


def get_element_count(the_list):
    count = sum(len(x) for x in the_list)
    return count


def flat_list(the_list):
    return [item for sublist in the_list for item in sublist]


def get_narrower_module(channel_range, module_config):
    len_list = []
    for i in module_config:
        len_list.append(len(i))
    count = get_element_count(module_config)
    config_list = get_search_space(count, channel_range)
    config_array = np.array(config_list)
    module_config_array = np.array(flat_list(module_config))
    equal_module_config_array = config_array <= module_config_array
    equal_module_config_array = np.prod(equal_module_config_array, 1)
    index = np.where(equal_module_config_array == 1)
    narrower_config = config_array[index[0]]
    narrower_config = narrower_config.tolist()
    result = []
    for i in narrower_config:
        result.append([i[:len_list[0]], i[len_list[0]:len_list[1] + len_list[0]], i[len_list[1] + len_list[0]:]])
    return remove_repeated_element(result)


def get_latency(module, input_size):
    module_input = torch.randn(input_size)
    start = time.time()
    output = module(module_input)
    end = time.time()
    return end - start


def get_excellent_module(trained_module):
    excellent_module = np.empty(shape=(0, 3))
    acc_and_lat = trained_module[:, 1:]
    for module in trained_module:
        tmp = copy.deepcopy(acc_and_lat)
        tmp[:, 0] = acc_and_lat[:, 0] <= module[1]
        tmp[:, 1] = acc_and_lat[:, 1] >= module[2]
        tmp = np.sum(tmp, axis=1)
        if 0 not in tmp:
            excellent_module = np.append(excellent_module, [module], axis=0)
    return excellent_module

a = get_all_search_space(3,5,[4,8,16,32,64,128])
b = get_search_space(4,[4,8,16,32,64,128])

