from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)


def build_model(cfg):
    """Build model."""
    return MODELS.build(cfg)
