from data.data_loader import (
    Dataset_MTS,
    Dataset_MTS_NPY,
    Dataset_DAN_Watershed,
    Dataset_Reservoir,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
)
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
    'Lexington': Dataset_Reservoir,
    'Ross_noRain': Dataset_DAN_Watershed,
    'Ross_S_fixed': Dataset_DAN_Watershed,
    'Saratoga_S_fixed': Dataset_DAN_Watershed,
    'SFC_S_fixed': Dataset_DAN_Watershed,
    'UpperPen_S_fixed': Dataset_DAN_Watershed,
    'Saratoga_noRain': Dataset_DAN_Watershed,
    'SFC_noRain': Dataset_DAN_Watershed,
    'UpperPen_noRain': Dataset_DAN_Watershed,
    'Ross': Dataset_DAN_Watershed,
    'Saratoga': Dataset_DAN_Watershed,
    'SFC': Dataset_DAN_Watershed,
    'UpperPen': Dataset_DAN_Watershed,
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
    if Data is Dataset_DAN_Watershed:
        data_kwargs.update(
            dan_norm_type=args.dan_norm_type,
        )
    elif Data is Dataset_MTS_NPY:
        data_kwargs.update(
            norm_type=getattr(args, 'norm_type', 'std'),
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
