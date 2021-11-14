import random
import torch
import os
import numpy as np

from sklearn.metrics import roc_auc_score, log_loss


def set_seed(seed):
    """
    Set seed to everything and turn on cudnn.deterministic so results can be reproduced.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_net(model, output_path, name, fold, stage, epoch):
    """
    Save full model to file.
    """
    output_file = '{}model/{}_{}_{}_{}.pth'.format(output_path, name, fold, stage, epoch)
    torch.save(model, output_file)
    return output_file


def calc_logloss(targets, outputs, eps=1e-5):
    """
    Calculate weighted log loss. This should return results similar to Kaggle metric.
    """
    try:
        logloss_classes = [log_loss(np.floor(targets[:, i]), np.clip(outputs[:, i], eps, 1 - eps)) for i in range(6)]
    except ValueError as e:
        logloss_classes = [1, 1, 1, 1, 1, 1]

    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[1, 1, 1, 1, 1, 2]),
    }


def calc_auc(targets, outputs):
    """
    Calculate micro and macro roc auc score.
    """
    macro = roc_auc_score(np.floor(targets), outputs, average='macro')
    micro = roc_auc_score(np.floor(targets), outputs, average='micro')
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }
