from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import PathConfig, ParamConfig


def get_mnist_loader():
    training_data = datasets.MNIST(
        root=PathConfig.path_root_images,
        train=True,
        download=False,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root=PathConfig.path_root_images,
        train=False,
        download=False,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data,
                                  batch_size=ParamConfig.size_train_batch)
    test_dataloader = DataLoader(test_data,
                                 batch_size=ParamConfig.size_test_batch)

    return train_dataloader, test_dataloader
