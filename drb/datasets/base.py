from abc import abstractmethod

import torch
from mmcv.parallel import DataContainer as DC


class Base(torch.utils.data.Dataset):
    def __init__(self,
                 *args,
                 **kw_args) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        img_meta = {'ori_shape': (224, 224)}
        data = dict(img=torch.randn(
            (3, 224, 224), dtype=torch.float32) + (index % 100),
                    img_metas=DC(img_meta, cpu_only=True))
        data['target'] = torch.tensor([index % 100], dtype=torch.float32)
        return data

    def pre_evaluate(self, predicts, inputs):
        return [{'mse': 0}]

    def evaluate(self, results, logger, **kw_args):
        mse = sum([_['mse'] for _ in results]) / len(results)
        eval_results = dict(mse=mse)
        return eval_results
