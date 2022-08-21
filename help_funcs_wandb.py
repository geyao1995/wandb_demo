from pathlib import Path

from config import ParamConfig
import torch
import wandb
from dataclasses import asdict
import os


def set_wandb_env(is_online: bool = True):
    os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'  # ignore *.pth files
    if is_online:
        os.environ['WANDB_MODE'] = 'online'
    else:
        # can be later synced with the `wandb sync` command.
        os.environ['WANDB_MODE'] = 'offline'


def define_wandb_lr_metrics():
    batch_step = 'batch_step'
    train_lr = 'train/lr'

    wandb.define_metric(batch_step)
    wandb.define_metric(train_lr, step_metric=batch_step)

    return batch_step, train_lr


def make_wandb_config(my_config: ParamConfig):
    wandb_config = asdict(my_config)

    return wandb_config


def convert_wandb_config(wandb_config):
    """convert wandb config to dataclass config"""
    param_config = ParamConfig(**wandb_config)
    return param_config


def save_model_to_wandb_dir(model: torch.nn.Module, idx_epoch: int):
    weights_path = Path(wandb.run.dir).joinpath('weights')
    weights_path.mkdir(exist_ok=True, parents=False)
    torch.save(model.state_dict(), weights_path.joinpath(f'epoch_{idx_epoch}.pth'))
    print(f'Save epoch {idx_epoch} model weights to {weights_path}')


def get_run_obj(run_instance):
    """
    :param run_instance: string, <entity>/<project>/<run_id>
    :return: run object
    """
    # https://docs.wandb.ai/ref/python/public-api/api
    api = wandb.Api()
    run = api.run(run_instance)

    return run


def export_wandb_run_data(run_instance, keys):
    """
    :param run_instance: string, <entity>/<project>/<run_id>
    :param is_from_cloud: bool
    :return:
    """
    run = get_run_obj(run_instance)

    if run.state == "finished":
        # https://github.com/wandb/wandb/blob/latest/wandb/apis/public.py#L1968
        return run.history(keys=keys, pandas=False)
    else:
        print('Run is not finished!')


def export_wandb_run_config(run_instance):
    run = get_run_obj(run_instance)

    if run.state == "finished":
        return run.config
    else:
        print('Run is not finished!')


def export_files_name(run_instance):
    run = get_run_obj(run_instance)

    if run.state == "finished":
        return [i for i in run.files()]
        # download
        # for file in run.files():
        #     file.download()
    else:
        print('Run is not finished!')


if __name__ == '__main__':
    run_instance = 'geyao/my-mnist-test-project/3a21s1op'
    r = export_wandb_run_data(run_instance, keys=['train-phase/loss', ])
    print(r)
