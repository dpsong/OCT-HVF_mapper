import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .base import Base
from .builder import DATASETS


TOTAL_OCT_SLICES = 256
OCT_IMAGE_SIZE = (224, 224)
GRAYSCALE_IMAGENET_MEAN = sum(IMAGENET_DEFAULT_MEAN) / 3.0
GRAYSCALE_IMAGENET_STD = sum(IMAGENET_DEFAULT_STD) / 3.0


def get_vf_info(filename: str) -> int:
    with open(filename, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            match = re.match(r'^\s*Age:?\s+(\d+)\b', line)
            if match:
                return int(match.group(1))
    raise ValueError(f'Failed to find Age in VF metadata file: {filename}')


@DATASETS.register_module()
class OCTVFDataset(Base):

    @staticmethod
    def _resolve_path(data_root: str, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(data_root, path)

    @staticmethod
    def _resolve_existing_path(data_root: str, path: str) -> str:
        if os.path.isabs(path) or os.path.exists(path):
            return path
        return os.path.join(data_root, path)

    @staticmethod
    def _load_oct_frame(frame_path: str) -> np.ndarray:
        with Image.open(frame_path) as image:
            frame = image.convert('L').resize(OCT_IMAGE_SIZE)
            return np.array(frame, copy=False)

    def __init__(self,
                 data_root,
                 split=None,
                 num_classes=None,
                 slice_dim=8,
                 num_thread=1,
                 reg_targets_dim=54,
                 cls_targets_dim=52,
                 mode='train',
                 oct_path=None,
                 age=None,
                 test_age=None,
                 dataAug=False,
                 *args,
                 **kw_args) -> None:
        super().__init__(*args, **kw_args)
        if num_classes is None:
            raise ValueError('num_classes must be provided.')
        if not 1 <= slice_dim <= TOTAL_OCT_SLICES:
            raise ValueError(
                f'slice_dim must be in [1, {TOTAL_OCT_SLICES}], got {slice_dim}.')
        if split is not None and oct_path is not None:
            raise ValueError('split and oct_path cannot be provided at the same time.')
        if split is None and oct_path is None:
            raise ValueError('Either split or oct_path must be provided.')

        self._data_root = data_root
        self.dataAug = dataAug
        self.data_aug = dataAug
        self.slice_num = slice_dim
        self.reg_targets_dim = reg_targets_dim
        self.cls_targets_dim = cls_targets_dim
        self.mode = mode
        self.test_age = test_age
        self._num_thread = num_thread
        self._threadpool: Optional[ThreadPoolExecutor] = None
        self.num_classes = num_classes
        self._db: List[Dict[str, object]] = []

        if split is not None:
            self._db.extend(self._load_split_samples(split))
        else:
            self._db.append(self._build_oct_path_sample(oct_path, age))

    def __len__(self):
        return len(self._db)

    def _build_oct_frames(self, oct_dir: str) -> List[str]:
        oct_dir = self._resolve_path(self._data_root, oct_dir)
        if not os.path.isdir(oct_dir):
            raise FileNotFoundError(f'Failed to find OCT directory: {oct_dir}')

        oct_frames = [
            os.path.join(oct_dir, f'slice_{index}.png')
            for index in range(TOTAL_OCT_SLICES)
        ]
        missing_frames = [frame for frame in oct_frames if not os.path.exists(frame)]
        if missing_frames:
            raise FileNotFoundError(
                f'Missing OCT slice file: {missing_frames[0]}. '
                f'Expected {TOTAL_OCT_SLICES} slices in {oct_dir}.')
        return oct_frames

    def _resolve_sample_age(self, vf_path: str) -> int:
        vf_txt_path = vf_path.replace('.json', '.txt')
        if os.path.exists(vf_txt_path):
            return get_vf_info(vf_txt_path)
        if self.mode == 'test' and self.test_age is not None:
            return int(self.test_age)
        raise FileNotFoundError(
            f'Failed to find VF metadata file: {vf_txt_path}. '
            'Provide the matching .txt file or set test_age for test mode.')

    def _load_split_samples(self, split: str) -> List[Dict[str, object]]:
        split_path = self._resolve_existing_path(self._data_root, split)
        with open(split_path, 'r', encoding='utf-8') as file_obj:
            split_data = json.load(file_obj)

        samples = []
        for vf_path, oct_path in split_data.items():
            vf_path = self._resolve_path(self._data_root, vf_path)
            samples.append({
                'oct_frames': self._build_oct_frames(oct_path),
                'vf': vf_path,
                'age': self._resolve_sample_age(vf_path),
            })
        return samples

    def _build_oct_path_sample(self, oct_path: str, age: Optional[int]) -> Dict[str, object]:
        if self.mode != 'test':
            raise ValueError('oct_path inference is only supported when mode="test".')
        if age is None:
            raise ValueError('age must be provided when using oct_path inference.')
        return {
            'oct_frames': self._build_oct_frames(oct_path),
            'vf': None,
            'age': int(age),
        }

    def _sample_oct_frames(self, oct_frames: List[str]) -> List[str]:
        if self.slice_num >= TOTAL_OCT_SLICES:
            return oct_frames

        step = len(oct_frames) // self.slice_num
        if step <= 0:
            raise ValueError(
                f'Invalid OCT sampling step for {len(oct_frames)} frames and '
                f'slice_num={self.slice_num}.')

        points = np.arange(0, len(oct_frames), step)
        if self.data_aug:
            indices = [int(points[i] + np.random.randint(step)) for i in range(self.slice_num)]
        else:
            indices = [int(points[i] + step / 2) for i in range(self.slice_num)]
        return [oct_frames[index] for index in indices]

    def _load_oct_tensor(self, oct_frames: List[str]) -> torch.Tensor:
        if self._threadpool is None:
            self._threadpool = ThreadPoolExecutor(max_workers=self._num_thread)

        tasks = [
            self._threadpool.submit(self._load_oct_frame, frame_path)
            for frame_path in oct_frames
        ]
        oct_frames_np = [task.result() for task in tasks]
        oct_tensor = torch.stack(
            [torch.from_numpy(frame.copy()).to(torch.float32) for frame in oct_frames_np],
            dim=0)

        if self.data_aug:
            aug_noise = (torch.randn_like(oct_tensor) * 0.1).clamp(-0.25, 0.25)
            aug_mask = torch.empty_like(oct_tensor).uniform_() <= 0.2
            oct_tensor = oct_tensor / 255.0 + aug_noise * aug_mask
        else:
            oct_tensor = oct_tensor / 255.0

        oct_tensor = (oct_tensor - GRAYSCALE_IMAGENET_MEAN) / GRAYSCALE_IMAGENET_STD
        return oct_tensor.unsqueeze(0)

    def _load_vf_targets(self, vf_path: str):
        with open(vf_path, 'r', encoding='utf-8') as file_obj:
            vf_data = json.load(file_obj)

        vf_num = np.asarray(vf_data['Sensitivity'], dtype=np.float32)
        vf_num = (vf_num - 20.0) / 40.0

        vf_ppd = np.asarray(vf_data['PDP'], dtype=np.float32)
        if self.num_classes == 2:
            vf_ppd[vf_ppd > 1] = 1

        cls_indices = (vf_ppd > 0).astype(np.int64)
        if self.reg_targets_dim == 54:
            reg_indices = np.hstack(
                (cls_indices[:19], -1, cls_indices[19:27], -1, cls_indices[27:]))
        else:
            reg_indices = cls_indices

        weight_indices = torch.from_numpy(
            np.hstack((reg_indices, cls_indices)).astype(np.int64))
        vf_target = torch.from_numpy(
            np.hstack((vf_num, vf_ppd)).astype(np.float32))
        return weight_indices, vf_target

    def __getitem__(self, idx):
        sample = self._db[idx]
        age = sample['age']
        oct_frames = self._sample_oct_frames(sample['oct_frames'])
        oct_tensor = self._load_oct_tensor(oct_frames)

        if self.mode == 'test':
            weight_indices = torch.tensor([-1], dtype=torch.int64)
            vf_target = torch.tensor([-1], dtype=torch.float32)
        else:
            weight_indices, vf_target = self._load_vf_targets(sample['vf'])

        return dict(
            img=oct_tensor,
            target=vf_target,
            weight_indices=weight_indices,
            age=age,
            idx=idx,
            img_metas=DC({}, cpu_only=True))

    @torch.no_grad()
    def pre_evaluate(self, predicts, inputs):
        device = predicts['reg'].device
        reg_target = inputs['target'][:, 0 : self.reg_targets_dim]
        cls_target = inputs[
            'target'][:, self.reg_targets_dim : self.reg_targets_dim + self.cls_targets_dim]
        reg_target = reg_target.contiguous().view(-1).to(device)
        cls_target = cls_target.contiguous().view(-1).to(device)

        reg_pred = predicts['reg'] * 40 + 20
        reg_target = reg_target * 40 + 20
        reg_pred = reg_pred.view(-1)
        _, cls_pred = torch.max(predicts['cls'].data.view(-1, self.num_classes), 1)
        reg_abs_sum = (reg_pred - reg_target).abs().sum().item()
        reg_count = reg_target.numel()
        cls_correct = (cls_pred == cls_target).sum().item()
        cls_count = cls_target.numel()

        return [{
            'reg_abs_sum': reg_abs_sum,
            'reg_count': reg_count,
            'cls_correct': cls_correct,
            'cls_count': cls_count,
        }]

    @torch.no_grad()
    def evaluate(self, results, logger, **kw_args):
        reg_abs_sum = sum(result['reg_abs_sum'] for result in results)
        reg_count = sum(result['reg_count'] for result in results)
        cls_correct = sum(result['cls_correct'] for result in results)
        cls_count = sum(result['cls_count'] for result in results)
        if reg_count == 0 or cls_count == 0:
            raise ValueError('Evaluation received empty targets.')

        mad = reg_abs_sum / reg_count
        acc = cls_correct / cls_count
        eval_results = dict(mad=mad, acc=acc)
        print_log(f'val: mad={mad}, acc={acc}', logger)
        return eval_results


if __name__ == '__main__':
    data_root = 'data/example/'
    split = 'data/TrainVal_split/train_split.json'
    test_dataset = OCTVFDataset(data_root=data_root, split=split, num_classes=2)
    print(test_dataset.__len__())
