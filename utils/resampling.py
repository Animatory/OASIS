import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader


def resample_dataset(opt, dataset_train, ious_per_class):
    label_counter = dataset_train.label_counter
    ious_per_class = ious_per_class[-label_counter.shape[1]:]
    ious_per_class = ious_per_class / ious_per_class.sum()

    weight_per_sample = np.reciprocal(label_counter @ ious_per_class + 1e-6)
    weight_per_sample = weight_per_sample / weight_per_sample.sum()

    samples_weight = torch.clip(torch.from_numpy(weight_per_sample).double(), min=1e-6)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    dataloader = DataLoader(dataset_train, batch_size=opt.batch_size, sampler=sampler,
                            num_workers=8, persistent_workers=True)
    return dataloader
