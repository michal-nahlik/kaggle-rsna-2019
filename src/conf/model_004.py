import conf.base_config_1 as base_conf


def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.model.source = 'efficientnet'
    cfg.model.name = 'efficientnet-b2'
    cfg.model.out_file_name = 'efficientnet-b2'
    return cfg


def get_stage_1_conf(cfg):
    cfg = base_conf.get_stage_1_conf(cfg)
    cfg.model.n_epochs = 0
    return  cfg


def get_stage_2_conf(cfg):
    cfg = base_conf.get_stage_2_conf(cfg)
    cfg.model.n_epochs = 8

    cfg.model.scheduler = dict(
        name='MultiStepLR',
        params=dict(
            milestones=[1, 3, 5, 7],
            gamma=3/4,
        )
    )

    return cfg


def get_stage_3_conf(cfg):
    cfg = base_conf.get_stage_3_conf(cfg)
    cfg.model.n_epochs = 0
    return cfg
