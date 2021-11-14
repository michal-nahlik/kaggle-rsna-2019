import conf.base_config_1 as base_conf


def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.model.source = 'pretrainedmodels'
    cfg.model.name = 'resnet34'
    cfg.model.out_file_name = 'resnet34'
    return cfg


def get_stage_1_conf(cfg):
    return base_conf.get_stage_1_conf(cfg)


def get_stage_2_conf(cfg):
    return base_conf.get_stage_2_conf(cfg)


def get_stage_3_conf(cfg):
    cfg = base_conf.get_stage_3_conf(cfg)
    cfg.model.n_epochs = 4
    return cfg
