import wandb
from config import ParamConfig
from get_loader import get_mnist_loader
from help_funcs import set_seed, get_device, get_lr_scheduler
from model import FlattenModel
from tester import Tester
from trainer import Trainer
from help_funcs_wandb import set_wandb_env, make_wandb_config, from_wandb_config, save_model_to_wandb_dir
import torch.optim as optim
import pprint


def define_wandb_metrics():
    epoch_step = 'epoch_step'
    train_loss = 'train/loss'
    test_acc = 'test/acc'
    wandb.define_metric(epoch_step)
    wandb.define_metric(train_loss, step_metric=epoch_step, summary='min')
    wandb.define_metric(test_acc, step_metric=epoch_step, summary='max')

    return epoch_step, train_loss, test_acc


def train_model():
    set_wandb_env(is_online=True)
    param_config = make_wandb_config(ParamConfig())
    run = wandb.init(project="my-mnist-test-project", reinit=True,
                     group='Group-A', job_type='Train',
                     notes='This is a demo',
                     config=param_config)  # can be later synced with the `wandb sync` command.
    wandb.run.name = f'MNIST_test-{wandb.run.id}'
    wandb.run.log_code(".")  # walks the current directory and save files that end with .py.
    config_param = from_wandb_config(wandb.config)  # for parameter sweep
    print(f'Param config = \n'
          f'{pprint.pformat(config_param, indent=4)}')
    epoch_step, train_loss, test_acc = define_wandb_metrics()

    set_seed(config_param.seed)
    device = get_device()

    model = FlattenModel()
    model.to(device)

    train_loader, test_loader = get_mnist_loader()

    if config_param.lr_schedule == 'step':
        lr_init = config_param.lr_max
    elif config_param.lr_schedule == 'cyclic':
        lr_init = config_param.lr_min
    else:
        raise NotImplementedError
    optimizer = optim.SGD(model.parameters(), lr=lr_init)

    lr_steps = config_param.epoch_total * len(train_loader)
    lr_scheduler = get_lr_scheduler(optimizer, config_param.lr_schedule, lr_steps,
                                    config_param.lr_min, config_param.lr_max)

    trainer = Trainer(device, model, config_param, train_loader, optimizer, lr_scheduler)
    tester = Tester(device, model, test_loader)

    acc_all = []

    for idx_epoch in range(1, config_param.epoch_total + 1):
        loss_train, lr = trainer.train_epoch(idx_epoch)
        if idx_epoch % 1 == 0:
            acc = tester.evaluate()
            acc_all.append(acc)
            wandb.log({test_acc: acc, epoch_step: idx_epoch})
            save_model_to_wandb_dir(model, idx_epoch)

        wandb.log({train_loss: loss_train, epoch_step: idx_epoch})

    run.finish()


if __name__ == '__main__':
    train_model()
