import argparse
import os
import os.path as osp
import time
import warnings
import cv2
import numpy as np

import torch
import mmcv
from mmcv.utils import Config
from mmcv.runner import init_dist, get_dist_info
from mmcv.cnn.utils import revert_sync_batchnorm

from drb.apis import single_gpu_test, multi_gpu_test
from drb.models import build_model
from drb.datasets import build_dataset, build_dataloader
from drb import __version__


def parse_args():
    parser = argparse.ArgumentParser(description='Val a model')
    parser.add_argument('--data_root',
                        type=str,
                        required=True,
                        help='the test oct data path')
    parser.add_argument('--data_split',
                        type=str,
                        required=True,
                        help='the test oct data path')
    parser.add_argument('--load-from',
                        type=str,
                        required=True,
                        help='the checkpoint file to load weights from')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def test(model, dataset, num_classes=2, device='cuda'):
    results = []
    model.eval().to(device)
    for idx in range(len(dataset)):
        sample = dataset._db[idx]
        inputs = dataset[idx]
        img = inputs['img'].unsqueeze(0).to(device)
        age = torch.tensor([inputs['age']], dtype=torch.int64).to(device)
        with torch.no_grad():
            predicts = model(return_loss=False, img=img, age=age)
            reg_preds = predicts['reg'].cpu() * 40 + 20
            cls_output = predicts['cls'].data.view(-1, num_classes)
            _, cls_preds = torch.max(cls_output, 1)
            print(reg_preds)
            results.append({'sensitivity ': reg_preds.cpu().numpy(), 'pdp': cls_preds.cpu().numpy()})
    return results
            
                        
def main():
    args = parse_args()

    model_config = dict(type='OCTEVA3D', reg_targets_dim=54, cls_targets_dim=52, slice_dim=16, num_classes=2, grad_checkpointing=True)
    model = build_model(model_config)
    checkpoint = torch.load(args.load_from)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    test_data_config=dict(type='OCTVFDataset',
             data_root=args.data_root,
             split=args.data_split,
             reg_targets_dim=54,
             cls_targets_dim=52,
             num_classes=2,
             slice_dim=16,
             mode='test')

    dataset = build_dataset(test_data_config)    
    results = test(model, dataset, num_classes=2, device='cuda')


if __name__ == '__main__':
    main()
