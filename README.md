# OCT-HVF_mapper: Predicting VF PDP from OCT

This repository is the official implementation of the paper *Predicting visual field pattern deviation probability maps from volumetric OCT scans.*

## Install

1. Clone this repository and navigate to OCT-HVF_mapper folder
```bash
git clone https://github.com/dpsong/OCT-HVF_mapper.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n oct2vf python=3.8 -y
conda activate oct2vf
pip install -r requirements.txt
```

## Train

1. Prepare data

Laterality was handled by flipping the images and corresponding VF data for right eyes (OD) to align with the left eye (OS) orientation. This standardization facilitates the model's learning process by maintaining consistent anatomical landmarks across all data.

2. Start training!

```Shell
PYTHONPATH=. TYPE=locale GPUS=1 ./tools/train.sh configs/train_config.py 
```

## Eval

```Shell
PYTHONPATH=. python tools/test_demo.py --data_root ./data/ --data_split test_data.json --load-from ./checkpoint/
```
