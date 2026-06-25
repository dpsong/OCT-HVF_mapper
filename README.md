# OCT-HVF_mapper: Predicting VF PDP from OCT

This repository is the official implementation of the paper *Predicting visual field pattern deviation probability maps from volumetric OCT scans.*

## Install

1. Clone this repository and navigate to OCT-HVF_mapper folder
```bash
git clone https://github.com/dpsong/OCT-HVF_mapper.git
```

2. Install Package
```Shell
conda create -n oct2vf python=3.8 -y
conda activate oct2vf
pip install -r requirements.txt
```

## Train

1. Prepare data

The repository expects paired OCT and VF files under a shared `data_root`. A runnable example is provided in [data/example](/Users/diping/Workspace/OCT-HVF_mapper/data/example).

Expected directory layout:

```text
data/example/
├── oct/
│   ├── 000001/101754_slices/slice_0.png ... slice_255.png
│   ├── 000002/110159_slices/slice_0.png ... slice_255.png
│   └── 000003/73921_slices/slice_0.png ... slice_255.png
└── vf/
    ├── 000001/000001.txt
    ├── 000001/000001.json
    ├── 000002/000002.txt
    ├── 000002/000002.json
    ├── 000003/000003.txt
    └── 000003/000003.json
```

Each OCT volume is stored as 256 grayscale slice images named `slice_0.png` to `slice_255.png`. In the bundled examples:

- `data/example/oct/000001/101754_slices/` contains 256 slices and pairs with `data/example/vf/000001/000001.*`
- `data/example/oct/000002/110159_slices/` contains 256 slices and pairs with `data/example/vf/000002/000002.*`
- `data/example/oct/000003/73921_slices/` contains 256 slices and pairs with `data/example/vf/000003/000003.*`

The VF `.txt` file is used to read metadata such as age. For example:

```text
Eye: Left
Age: 24
MD: 0.02 dB
PSD: 1.52 dB
VFI: 99%
```

The VF `.json` file contains the training targets:

```json
{
  "Sensitivity": [54 values],
  "PDP": [52 values]
}
```

Split files map VF json paths to OCT directories, both relative to `data_root`. The provided examples are:

```json
{
  "vf/000001/000001.json": "oct/000001/101754_slices/",
  "vf/000003/000003.json": "oct/000003/73921_slices/"
}
```

```json
{
  "vf/000002/000002.json": "oct/000002/110159_slices/"
}
```

Laterality flipping is not performed automatically by this repository. Right-eye scans and their corresponding VF targets should be flipped offline first so that all samples match left-eye orientation. The bundled example files are already standardized this way, and their `.txt` files show `Eye: Left`.

2. Start training!

```Shell
PYTHONPATH=. TYPE=locale GPUS=1 ./tools/train.sh configs/train_config.py 
```

## Inference

`tools/test_demo.py` supports two inference modes.

Single-volume inference by providing the OCT slice directory and patient age:

```Shell
PYTHONPATH=. python tools/test_demo.py --data_root ./data/example/ --oct_path oct/000002/110159_slices/ --age 24 --load-from ./checkpoint/<checkpoint>.pth
```

Batch inference by providing a `data_split` file. In this mode, the script loads age from the corresponding VF `.txt` files:

```Shell
PYTHONPATH=. python tools/test_demo.py --data_root ./data/example/ --data_split ./data/TrainVal_split/val_split.json --load-from ./checkpoint/<checkpoint>.pth
```
