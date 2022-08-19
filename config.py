from dataclasses import dataclass

from pathlib import Path


@dataclass
class PathConfig:
    path_root_images = Path('/Users/geyao/torchvision_dataset')


@dataclass
class ParamConfig:
    dataset_name: str = 'MNIST'
    model_architecture: str = 'flatten'

    seed: int = 0

    epoch_total: int = 10
    size_train_batch: int = 128
    size_test_batch: int = 128

    lr_schedule: str = 'step'  # 'step', 'cyclic'
    lr_min: float = 0.001
    lr_max: float = 0.1

    def __post_init__(self):
        assert self.lr_schedule in ['step', 'cyclic']
        assert self.lr_min < self.lr_max
