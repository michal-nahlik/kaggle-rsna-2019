import numpy as np
import torch
import torch.nn
import net.factory as net_factory
import net.util as net_util

from apex import amp
from tqdm import tqdm


def train(cfg, model, data_loader_train, data_loader_val):

    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }

    criterion = net_factory.get_loss(cfg.model)
    optim = net_factory.get_optim(cfg.model, model.parameters())

    if cfg.model.scheduler is not None:
        scheduler = net_factory.get_scheduler(cfg.model, optim, best['epoch'])
    else:
        scheduler = None

    if cfg.model.use_amp:
        amp.initialize(model, optim, opt_level='O1')

    for epoch in range(best['epoch'] + 1, cfg.model.n_epochs):

        if cfg.model.balance_data:
            data_loader_train.dataset.balance_data(cfg.model.rus_seed + epoch)

        run_nn(cfg.model, 'train', model, data_loader_train,
               criterion=criterion, optim=optim, scheduler=scheduler, apex=cfg.model.use_amp)

        with torch.no_grad():
            val = run_nn(cfg.model, 'valid', model, data_loader_val, criterion=criterion)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }

        if val['loss'] <= best['loss']:
            best.update(detail)

        print('[best] ep: {} loss: {} score: {}'.format(best['epoch'], best['loss'], best['score']))

        net_util.save_net(model, cfg.output_path, cfg.model.out_file_name, cfg.fold, cfg.model.stage, epoch)

        if cfg.model.scheduler is not None and cfg.model.scheduler['name'] != 'OneCycleLR':
            scheduler.step()


def run_nn(cfg, mode, model, loader, criterion=None, optim=None, apex=None, scheduler=None):

    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise

    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for batch in tqdm(loader):

        ids = batch['image_name']
        inputs = batch['image']
        inputs = inputs.to(cfg.device, dtype=torch.float)

        outputs = model(inputs)

        if mode in ['train', 'valid']:
            targets = batch['labels']
            targets = targets.to(cfg.device, dtype=torch.float)

            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())
                targets_all.extend(targets.cpu().numpy())

        if mode in ['train']:
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()  # accumulate loss

            optim.step()  # update
            optim.zero_grad()  # flush

            if scheduler is not None and cfg.scheduler['name'] == 'OneCycleLR':
                scheduler.step()

        with torch.no_grad():
            ids_all.extend(ids)
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())

    result = {
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / len(loader),
    }

    if mode in ['train', 'valid']:
        result.update(net_util.calc_auc(result['targets'], result['outputs']))
        result.update(net_util.calc_logloss(result['targets'], result['outputs']))
        result['score'] = result['logloss']

        print('==============={}==============='.format(mode))
        print('auc:{} micro:{} macro:{}'.format(result['auc'], result['auc_micro'], result['auc_macro']))
        print('{} {}'.format(result['logloss'], np.round(result['logloss_classes'], 6)))
    else:
        print('')

    return result
