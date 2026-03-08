from data.data_loader import Dataset_MTS, Dataset_MTS_NPY, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import Dataset, DataLoader
import torch
import numpy
import random
data_dict = {
    'ETTh1_labeled': Dataset_ETT_hour,
    'ETTh2_labeled': Dataset_ETT_hour,
    'ETTm1_labeled': Dataset_ETT_minute,
    'ETTm2_labeled': Dataset_ETT_minute,
    'Weather_labeled': Dataset_MTS,
    'Exchange_labeled': Dataset_MTS,
    'Coyote': Dataset_MTS_NPY,
    'Lexington': Dataset_MTS_NPY,
    # 'Solar': Dataset_Solar,
    # 'PEMS': Dataset_PEMS,
    # 'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
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
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    #     data_set = dataset_loader(
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         data_split=args.data_split
    #     )
    data_kwargs = dict(
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
    if Data is Dataset_MTS:
        data_kwargs.update(
            dataset_name=args.data,
            start_point=getattr(args, 'start_point', None),
            train_point=getattr(args, 'train_point', None),
            test_start=getattr(args, 'test_start', None),
            test_end=getattr(args, 'test_end', None),
            train_seed=getattr(args, 'train_seed', None),
            train_volume=getattr(args, 'train_volume', None),
            val_seed=getattr(args, 'val_seed', None),
            val_size=getattr(args, 'val_size', None),
            test_stride=getattr(args, 'test_stride', 16),
        )
    elif Data is Dataset_MTS_NPY:
        data_kwargs.update(
            norm_type=getattr(args, 'norm_type', 'std'),
            merge_to_series=getattr(args, 'merge_to_series', False),
        )
    data_set = Data(**data_kwargs)
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
