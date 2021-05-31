from torch.utils import data


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def get_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders." + dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print(f"Created {dataset_name}, size train: {len(dataset_train)}, size val: {len(dataset_val)}")

    # dataloader_train = data.DataLoader(
    #     dataset_train, batch_size=opt.batch_size,
    #     sampler=data_sampler(dataset_train, shuffle=True, distributed=opt.distributed),
    #     drop_last=True, num_workers=8, persistent_workers=True,
    # )
    # dataloader_val = data.DataLoader(
    #     dataset_val, batch_size=opt.batch_size,
    #     sampler=data_sampler(dataset_val, shuffle=False, distributed=opt.distributed),
    #     drop_last=True, num_workers=8, persistent_workers=True,
    # )

    dataloader_train = data.DataLoader(dataset_train, batch_size=opt.batch_size, persistent_workers=True,
                                       shuffle=True, drop_last=True, num_workers=12)
    dataloader_val = data.DataLoader(dataset_val, batch_size=opt.batch_size, persistent_workers=True,
                                     shuffle=False, drop_last=False, num_workers=12)

    return dataloader_train, dataloader_val, dataset_train, dataset_val
