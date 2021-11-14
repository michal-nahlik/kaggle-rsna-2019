import torch
import net.adam
from net import lr_scheduler


def get_optim(cfg, parameters):
    """
    Create optimizer based on cfg.optim['name'] and parameters
    :param cfg:
    :param parameters:
    :return:
    """
    optim = getattr(net.adam, cfg.optim['name'])(parameters, **cfg.optim['params'])
    return optim


def get_loss(cfg):
    """
    Create weighted torch.nn loss based on cfg.loss['name'] and cfg.loss['params']
    :param cfg:
    :return:
    """
    loss = getattr(torch.nn, cfg.loss['name'])(weight=torch.FloatTensor([1,1,1,1,1,2]).cuda(), **cfg.loss['params'])
    return loss


def get_scheduler(cfg, optim, last_epoch):
    """
    Create scheduler based on config.
    :param cfg:
    :param optim:
    :param last_epoch:
    :return:
    """
    if cfg.scheduler['name'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler['params'],
        )
        scheduler.last_epoch = last_epoch
    elif cfg.scheduler['name'] == 'OneCycleLR':
        scheduler = getattr(lr_scheduler, cfg.scheduler['name'])(
            optim,
            **cfg.scheduler['params'],
        )
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler['name'])(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler['params'],
        )
    return scheduler