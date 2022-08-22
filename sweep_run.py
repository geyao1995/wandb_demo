import pprint
import wandb
from train_mnist import train_model
import yaml


def get_sweep_config_from_yaml_file(f_path):
    with open(f_path) as f:
        try:
            my_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    return my_dict


def get_sweep_config():
    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'test-result/accuracy',
        'goal': 'maximize'
    }
    parameters_dict = {
        'lr_schedule': {
            'values': ['step', 'cyclic']
        },
        'epoch_total': {
            'values': [2, 4]
        },
    }
    project = 'my-mnist-test-project'
    name = 'MNIST-Sweep-Test'
    description = 'use python code config'

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict
    sweep_config['project'] = project
    sweep_config['name'] = name
    sweep_config['description'] = description

    return sweep_config


def perform_sweep(is_from_file, f_path=None, main_func=None):
    """
    :param is_from_file: use yaml file or python code to make config
    :param f_path: needed if is_from_file is True.
    :param main_func: needed if is_from_file is False.
     A function to call instead of the "program" specifed in the config.
    :return: None
    """
    if is_from_file:
        assert f_path is not None
        sweep_config = get_sweep_config_from_yaml_file(f_path)
        main_func = None
    else:
        assert main_func is not None
        sweep_config = get_sweep_config()

    print(f'Sweep config = \n'
          f'{pprint.pformat(sweep_config, indent=4)}')

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=main_func)


if __name__ == '__main__':
    """
    Reference: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=zQzooy61aCO6
    """
    perform_sweep(is_from_file=False,
                  f_path='./sweep_config.yaml',
                  main_func=train_model)
