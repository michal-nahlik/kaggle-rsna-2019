import conf.base_config_2 as base_conf

def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.model.source = 'pretrainedmodels'
    cfg.model.name = 'se_resnext50_32x4d'
    cfg.model.out_file_name = 'se_resnext50_32x4d_base2'
    return cfg


def get_stage_1_conf(cfg):
    cfg.model.init_lr = 1e-3
    cfg.model.stage = 2
    cfg.model.n_epochs = 2
    cfg.model.balance_data = True

    cfg.model.optim = dict(
        name='RAdam',
        params=dict(
            lr=cfg.model.init_lr,
        )
    )

    return cfg

def get_stage_2_conf(cfg):
    cfg.model.stage = 2
    cfg.model.n_epochs = 0
    return cfg

def get_stage_3_conf(cfg):
    cfg.model.stage = 3
    cfg.model.n_epochs = 5
    cfg.model.init_lr = 1e-4
    cfg.model.balance_data = False

    cfg.model.optim = dict(
        name='RAdam',
        params=dict(
            lr=cfg.model.init_lr,
        )
    )

    cfg.model.scheduler = dict(
        name='OneCycleLR',
        params=dict(
            max_lr=0.01,
            epochs=cfg.model.n_epochs,
            steps_per_epoch=8392,
        )
    )

    return cfg
