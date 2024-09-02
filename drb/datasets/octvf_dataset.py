import os
import json
from concurrent.futures import ThreadPoolExecutor
from posixpath import split
from mmcv.utils import print_log

from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score

from mmcv.parallel import DataContainer as DC
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as nnF
from .builder import DATASETS
from .base import Base
import torch.nn.functional as F

def get_vf_info(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Age' in line:
                age = int(line.strip().split()[1])
    return age


@DATASETS.register_module()
class OCTVFDataset(Base):

    def __init__(self,
                 data_root,
                 split,
                 num_classes,
                 slice_dim=8,
                 num_thread=1,
                 reg_targets_dim=54,
                 cls_targets_dim=52,
                 mode='train',
                 dataAug=False,
                 *args,
                 **kw_args) -> None:
        self._data_root = data_root
        self.dataAug = dataAug
        self.slice_num = slice_dim
        self.reg_targets_dim = reg_targets_dim
        self.cls_targets_dim = cls_targets_dim
        self.mode = mode
        self._db = list()

        
        with open(split) as fr:
            data = json.load(fr)
            for _, vf_path in enumerate(data):
                octpath = data[vf_path]
                vf_path = os.path.join(data_root, vf_path)
                if '/cpfs01/user/songdiping/Workspaces/' not in octpath:
                    octpath = os.path.join(data_root, octpath)
                oct_frames = list()
                for i in range(256):
                    frame = os.path.join(octpath, 'slice_{}.png'.format(i))
                    oct_frames.append(frame)
                age = get_vf_info(vf_path.replace('.json', '.txt'))
                sample = {'oct_frames': oct_frames, 'vf': vf_path, 'age':age}
                self._db.append(sample)
        self._num_thread = num_thread
        self._threadpool = None
        self.num_classes = num_classes

    def __len__(self):
        return len(self._db)

    def __getitem__(self, idx):
        if self._threadpool is None:
            self._threadpool = ThreadPoolExecutor(max_workers=self._num_thread)
        sample = self._db[idx]

        age = sample['age']
        oct_frames = sample['oct_frames']
        if self.slice_num < 256:
            step = int(len(oct_frames) / self.slice_num)
            points = np.arange(0, len(oct_frames), step)
            if self.dataAug:
                idxs = [points[i] + np.random.choice(step, 1) for i in range(self.slice_num)]
            else:
                idxs = [points[i] + int(step / 2) for i in range(self.slice_num)]
            oct_frames = [oct_frames[int(idxs[i])] for i in range(self.slice_num)]
        
        tasks = list()
        for p in oct_frames:
            future = self._threadpool.submit(
                    lambda p: Image.open(p).convert('L'), p)
            tasks.append(future)

        oct_frames = [
            np.array(_.result().resize((224, 224)), copy=False) for _ in tasks 
        ]

        # To tensor
        oct_frames = torch.cat(
            [torch.from_numpy(_.copy()).unsqueeze(0) for _ in oct_frames],
            dim=0)
        imagenet_mean = sum(IMAGENET_DEFAULT_MEAN) / 3.
        imagenet_std = sum(IMAGENET_DEFAULT_STD) / 3.
        
        if self.dataAug:
            oct_frames = (oct_frames / 255. +
                        (torch.randn(oct_frames.shape) * 0.1).clamp(-0.25, 0.25) *
                        (torch.empty(oct_frames.shape).uniform_() <= 0.2) -
                        imagenet_mean) / imagenet_std
        else:
            oct_frames = (oct_frames / 255. - imagenet_mean) / imagenet_std

        oct_frames.unsqueeze_(0)

        if self.mode=='test':
            weight_indices = torch.Tensor([-1])
            vf_target = torch.Tensor([-1])
        else:
            vf_path = sample['vf']
            vf_data = np.load(vf_path, allow_pickle=True).item()

            vf_num = np.array(vf_data['Sensitivity'])
            vf_num -= 20
            vf_num /= 40.0

            vf_ppd = np.array(vf_data['PDP'])
            if self.num_classes == 2:
                vf_ppd[vf_ppd > 1] = 1

            cls_indices = (vf_ppd > 0).astype(int)
            if self.reg_targets_dim == 54:
                reg_indices = np.hstack((cls_indices[:19],-1,cls_indices[19:27],-1,cls_indices[27:]))
            else:
                reg_indices = cls_indices
            weight_indices = torch.from_numpy(np.hstack((reg_indices, cls_indices)))
            vf_target = torch.from_numpy(np.hstack((vf_num, vf_ppd)))

        data = dict(img=oct_frames,
                    target=vf_target,
                    weight_indices=weight_indices,
                    age=age,
                    idx=idx,
                    img_metas=DC({}, cpu_only=True))
        return data


    @torch.no_grad()
    def pre_evaluate(self, predicts, inputs):
        device = predicts['reg'].device
        eval_results = list()
        reg_target = inputs['target'][:, 0 : self.reg_targets_dim]
        cls_target = inputs['target'][:, self.reg_targets_dim : self.reg_targets_dim + self.cls_targets_dim]
        reg_target = reg_target.contiguous().view(-1).to(
            device) 
        cls_target = cls_target.contiguous().view(-1).to(
            device) 

        reg_pred = predicts['reg'] * 40 + 20
        reg_target = reg_target * 40 + 20
        reg_pred = reg_pred.view(-1)
        _, cls_pred = torch.max(predicts['cls'].data.view(-1, self.num_classes),
                               1) 
        mad = (reg_pred - reg_target).abs().mean().item()
        acc = (cls_pred == cls_target).sum().item() / cls_target.size()[0]
        eval_results.append({
            'mad': mad,
            'acc': acc,
        })

        return eval_results

    @torch.no_grad()
    def evaluate(self, results, logger, **kw_args):
        mad = sum([_['mad'] for _ in results]) / len(results)
        acc = sum([_['acc'] for _ in results]) / len(results)
        eval_results = dict(mad=mad,
                            acc=acc)
        print_log(
            'val: mad={}, acc={}'.format(mad, acc), logger)
        return eval_results


if __name__ == '__main__':
    data_root = '/cpfs01/user/songdiping/Workspaces/Data/oct2vf/zoc3d-2022_resize384/'
    split = 'TrainVal_split/macula_val_split.json'
    test_dataset = OCTVFDataset(data_root, split, num_classes=5)
    print(test_dataset.__len__())
