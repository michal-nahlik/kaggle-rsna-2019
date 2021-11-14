import conf.base_config_1 as base_conf


def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.model.source = 'pretrainedmodels'
    cfg.model.name = 'se_resnext50_32x4d'
    cfg.model.out_file_name = 'se_resnext50_32x4d_2_2_2'
    return cfg


def get_stage_1_conf(cfg):
    cfg.model.init_lr = 8e-4
    cfg.model.stage = 1
    cfg.model.n_epochs = 2
    cfg.model.balance_data = False

    cfg.model.optim = dict(
        name='RAdam',
        params=dict(
            lr=cfg.model.init_lr,
        )
    )

    cfg.model.scheduler = dict(
        name='MultiStepLR',
        params=dict(
            milestones=[1],
            gamma=1/2,
        )
    )

    return cfg


def get_stage_2_conf(cfg):
    return base_conf.get_stage_2_conf(cfg)


def get_stage_3_conf(cfg):
    return base_conf.get_stage_3_conf(cfg)
