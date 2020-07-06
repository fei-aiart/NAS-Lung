import searchspace.res_search_space as res_search_space
import torch.nn as nn
import logging
import torch
import argparse

# set args
parser = argparse.ArgumentParser(description='searching')
# parser.add_argument('--sub', type=int, default=5, help="sub data set")
parser.add_argument('--fold', type=int, default=5, help="fold")
parser.add_argument('--gpu_id', type=str, default='0', help="gpu_id")
parser.add_argument('--lr', type=float, default=0.0002, help="lr")
parser.add_argument('--epoch', type=int, default=20, help="epoch")
parser.add_argument('--num_workers', type=int, default=20, help="num_workers")
parser.add_argument('--train_data_path', type=str, default='/data/xxx/LUNA/cls/crop_v3', help="train_data_path")
parser.add_argument('--test_data_path', type=str, default='/data/xxx/LUNA/rowfile/subset', help="test_data_path")
parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
parser.add_argument('--max_depth', type=int, default=9, help="max_depth")
parser.add_argument('--min_depth', type=int, default=3, help="min_depth")
parser.add_argument('--save_module_path', type=str, default='Module')
parser.add_argument('--log_file', type=str, default='log_search')

if __name__ == '__main__':
    args = parser.parse_args()
    fold = args.fold
    channel_range = [4, 8, 16, 32, 64, 128]
    batch_size = args.batch_size
    max_depth = args.max_depth
    min_depth = args.min_depth
    criterion = nn.CrossEntropyLoss()
    gpu_id = args.gpu_id

    input_shape = [1, 1, 32, 32, 32]
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    use_gpu = torch.cuda.is_available()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    lr = args.lr
    save_module_path = args.save_module_path
    num_workers = args.num_workers
    epoch = args.epoch
    # sub = args.sub
    # search model
    res_search = res_search_space.ResSearchSpace(channel_range, max_depth, min_depth, train_data_path, test_data_path, fold,
                                                 batch_size, logging, input_shape, use_gpu, gpu_id, criterion, lr,
                                                 save_module_path, num_works=num_workers, epoch=epoch)
    res_search.main_method()
