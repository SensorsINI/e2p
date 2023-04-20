"""
 @Time    : 29.03.22 15:11
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : data_loaders.py
 @Function:
 
"""
from torch.utils.data import DataLoader
# local modules
from train.data_loader.dataset import *
from train.utils.data import concatenate_subfolders, concatenate_datasets


class InferenceDataLoader(DataLoader):
    def __init__(self, data_path, num_workers=0, pin_memory=True, dataset_kwargs=None, ltype="H5", real_data=False, direction=None):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if ltype == "H5":
            if direction is None:
                # raw to raw
                # dataset = DynamicH5Dataset_v2e(data_path, **dataset_kwargs)
                # raw to p
                if not real_data:
                    dataset = DynamicH5Dataset_v2e_p(data_path, **dataset_kwargs)
                    # dataset = DynamicH5Dataset_v2e_s012_iad(data_path, **dataset_kwargs)
                else:
                    dataset = DynamicH5Dataset_v2e_p_real_data(data_path, **dataset_kwargs)
                    # dataset = DynamicH5Dataset_v2e_s012_iad_real_data(data_path, **dataset_kwargs)
            else:
                # extract events of four directions separately
                if direction == '90':
                    print("Dataloader is 90.")
                    dataset = DynamicH5Dataset_v2e_p_90(data_path, **dataset_kwargs)
                elif direction == '45':
                    print("Dataloader is 45.")
                    dataset = DynamicH5Dataset_v2e_p_45(data_path, **dataset_kwargs)
                elif direction == '135':
                    print("Dataloader is 135.")
                    dataset = DynamicH5Dataset_v2e_p_135(data_path, **dataset_kwargs)
                elif direction == '0':
                    print("Dataloader is 0.")
                    dataset = DynamicH5Dataset_v2e_p_0(data_path, **dataset_kwargs)
                else:
                    print('Direction wrong!')
        elif ltype == "MMP":
            dataset = MemMapDataset(data_path, **dataset_kwargs)
        else:
            raise Exception("Unknown loader type {}".format(ltype))
        super().__init__(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


class HDF5DataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


class MemMapDataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

