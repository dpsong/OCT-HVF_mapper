import argparse
import os
import torch

from drb.models import build_model
from drb.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference for OCT to VF')
    parser.add_argument('--data_root',
                        type=str,
                        default='',
                        help='root directory for split mode or relative oct paths')
    parser.add_argument('--data_split',
                        type=str,
                        help='json split file for dataset-based evaluation/inference')
    parser.add_argument('--oct_path',
                        type=str,
                        help='single OCT slice directory, e.g. oct/000002/110159_slices/')
    parser.add_argument('--age',
                        type=int,
                        help='patient age for single OCT inference')
    parser.add_argument('--load-from',
                        type=str,
                        required=True,
                        help='the checkpoint file to load weights from')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if args.data_split is None and args.oct_path is None:
        parser.error('one of --data_split or --oct_path is required')
    if args.data_split is not None and args.oct_path is not None:
        parser.error('--data_split and --oct_path are mutually exclusive')
    if args.oct_path is not None and args.age is None:
        parser.error('--age is required when using --oct_path')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def test(model, dataset, num_classes=2, device='cuda'):
    results = []
    model.eval().to(device)
    for idx in range(len(dataset)):
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

    if args.oct_path is not None:
        test_data_config = dict(type='OCTVFDataset',
                                data_root=args.data_root,
                                oct_path=args.oct_path,
                                age=args.age,
                                reg_targets_dim=54,
                                cls_targets_dim=52,
                                num_classes=2,
                                slice_dim=16,
                                mode='test')
        dataset = build_dataset(test_data_config)
        test(model, dataset, num_classes=2, device='cuda')
    else:
        test_data_config = dict(type='OCTVFDataset',
                                data_root=args.data_root,
                                split=args.data_split,
                                reg_targets_dim=54,
                                cls_targets_dim=52,
                                num_classes=2,
                                slice_dim=16,
                                mode='test')
        dataset = build_dataset(test_data_config)
        test(model, dataset, num_classes=2, device='cuda')


if __name__ == '__main__':
    main()
