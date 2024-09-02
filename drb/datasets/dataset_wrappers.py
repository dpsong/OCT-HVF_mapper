# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    support evaluation and formatting results

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the concatenated
            dataset results separately, Defaults to True.
    """
    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)

    @torch.no_grad()
    def pre_evaluate(self, predicts, inputs):
        self.datasets[0].pre_evaluate(predicts, inputs)

    @torch.no_grad()
    def evaluate(self, results, logger=None, **kw_args):
        self.datasets[0].evaluate(results, logger, **kw_args)


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return int(self.times * self._ori_len)

    @torch.no_grad()
    def pre_evaluate(self, predicts, inputs):
        self.dataset.pre_evaluate(predicts, inputs)

    @torch.no_grad()
    def evaluate(self, results, logger=None, **kw_args):
        self.dataset.evaluate(results, logger, **kw_args)