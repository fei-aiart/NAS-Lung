"""the width of a block to be no narrower than its preceding block"""

import numpy as np
from .search_space_utils import *
import random
from data_utils import *
from models.cnn_res import ConvRes


class ResSearchSpace:
    """
    search class
    """
    def __init__(self, channel_range, max_depth, min_depth, trained_data_path, test_data_path, fold, batch_size,
                 logging, input_shape, use_gpu, gpu_id, criterion, lr, save_module_path, num_works, epoch):
        self.lr = lr
        self.epoch = epoch
        self.num_works = num_works
        self.fold = fold
        # self.sub = sub
        self.save_module_path = save_module_path
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.criterion = criterion
        self.logging = logging
        self.input_shape = input_shape
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.max_depth = max_depth
        self.channel_range = channel_range
        self.trained_module_acc_lat = np.empty((0, 3))
        # self.trained_module_acc_lat = [module_config,acc,lat]
        self.pruned_module = []
        # initialize all architecture set
        self.untrained_module = get_all_search_space(min_len=min_depth, max_len=max_depth, channel_range=channel_range)
        # load data set
        self.train_loader, self.test_loader = load_data(trained_data_path, test_data_path, self.fold, batch_size,
                                                        self.num_works)
        self.trained_yw = []
        self.save_module_path = save_module_path
        self.trained_yw_and_module = []
        if not os.path.isdir(self.save_module_path):
            os.mkdir(self.save_module_path)

    def random_generate(self):
        """
        get untrained model config

        :return: model config
        """
        count = self.untrained_module.__len__()
        index = random.randint(0, count)
        return self.untrained_module[index]

    def main_method(self):
        """
        search main method
        """
        B = []
        stable_time = 0
        repeat_time = 0
        while True:
            # get untrained model
            config = self.random_generate()
            # config = [[512, 512, 512,512,512], [512, 512, 512,512,512], [512, 512, 512, 512,512]]
            self.untrained_module.remove(config)
            # train model
            net = ConvRes(config)
            net_lat = get_module_lat(net, input_shape=self.input_shape)
            net = net_to_cuda(net, use_gpu=self.use_gpu, gpu_ids=self.gpu_id)
            optimizer = optim.Adam(net.parameters(), lr=self.lr, betas=(0.5, 0.999))
            acc = get_acc(net, self.use_gpu, self.train_loader, self.test_loader, optimizer, self.criterion,
                          self.logging, self.lr, config, self.epoch)
            print(f'module:{config}\nacc:{acc} lat:{net_lat}')
            self.logging.info(f'config:{config}\nacc:{acc} lat:{net_lat}')
            del net
            self.trained_module_acc_lat = np.append(self.trained_module_acc_lat, [[config, acc, net_lat]], axis=0)
            # prune model
            for module in self.trained_module_acc_lat:
                yw = get_yw(self.trained_module_acc_lat, module)
                if len(yw) != 0:
                    module_config = module[0]
                    yw_config = yw[0]
                    yw_lat = yw[2]
                    if [yw_config, module_config] not in self.trained_yw_and_module:
                        print(f'yw:{yw_config}\nmodule:{module_config}')
                        self.logging.info(f'yw:{yw_config}\nmodule:{module_config}')
                        narrower_module = get_narrower_module(self.channel_range, module_config)
                        print('found narrower_module:' + str(narrower_module.__len__()))
                        self.logging.info(
                            'found narrower_module:' + str(narrower_module.__len__())
                        )
                        shallower_module = get_shallower_module(self.min_depth, [module_config], shallower_module=[])
                        print('found shallower_module:' + str(shallower_module.__len__()))
                        self.logging.info(
                            'found shallower_module:' + str(shallower_module.__len__())
                        )
                        pruned_narrower_module = 0
                        for i in narrower_module:
                            if i in self.untrained_module:
                                lat = get_latency(ConvRes(i), input_size=self.input_shape)
                                if lat > yw_lat:
                                    self.pruned_module.append(i)
                                    self.untrained_module.remove(i)
                                    pruned_narrower_module = pruned_narrower_module + 1
                        print('pruned_narrower_module:' + str(pruned_narrower_module))
                        self.logging.info(
                            'pruned_narrower_module:' + str(pruned_narrower_module)
                        )
                        pruned_shallower_module = 0
                        for i in shallower_module:
                            if i in self.untrained_module:
                                lat = get_latency(ConvRes(i), input_size=self.input_shape)
                                if lat > yw_lat:
                                    pruned_shallower_module = pruned_shallower_module + 1
                                    self.pruned_module.append(i)
                                    self.untrained_module.remove(i)
                        print('pruned_shallower_module:' + str(pruned_shallower_module))
                        self.logging.info(
                            'pruned_shallower_module:' + str(pruned_shallower_module)
                        )
                        self.trained_yw_and_module.append([yw_config, module_config])
                else:
                    print(f'{module[0]}yw not found')

            B1 = get_excellent_module(self.trained_module_acc_lat)
            if repeat_time % 20 == 0:
                np.save(self.save_module_path + '/' + str(repeat_time) + 'trained', self.trained_module_acc_lat)
                np.save(self.save_module_path + '/' + str(repeat_time) + 'excellent', B1)
            if np.array_equal(B1, B):
                stable_time = stable_time + 1
                B = copy.deepcopy(B1)
            else:
                stable_time = 0
                B = copy.deepcopy(B1)
            if stable_time > 30:
                break
            repeat_time = repeat_time + 1
