from data.data_loader import Dataset_MTS, Dataset_ETT_hour
from torch.utils.data import Dataset, DataLoader
import torch
import numpy
import random
data_dict = {
    'ETTh1_labeled': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'Solar': Dataset_Solar,
    # 'PEMS': Dataset_PEMS,
    # 'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    #     data_set = dataset_loader(
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         data_split=args.data_split
    #     )
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle,
    )
    print(flag, len(data_set))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )
    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    return data_set, data_loader


# from data.data_loader import Dataset_MTS
# from torch.utils.data import Dataset, DataLoader
# import torch
# import numpy
# import random
#
#
# def data_provider(args, flag):
#     dataset_loader = Dataset_MTS
#
#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = True
#         batch_size = args.batch_size
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size
#
#     data_set = dataset_loader(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         flag=flag,
#         size=[args.seq_len, args.label_len, args.pred_len],
#         data_split=args.data_split
#     )
#     print(flag, len(data_set))
#
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2 ** 32
#         numpy.random.seed(worker_seed)
#         random.seed(worker_seed)
#
#     g = torch.Generator()
#     g.manual_seed(0)
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last,
#         worker_init_fn=seed_worker,
#         generator=g,
#     )
#     return data_set, data_loader