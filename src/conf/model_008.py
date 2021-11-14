import conf.base_config_3 as base_conf


def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.model.source = 'torchvision'
    cfg.model.name = 'resnext50_32x4d'
    cfg.model.out_file_name = 'resnext50_32x4d_base3'
    return cfg


def get_stage_1_conf(cfg):
    return base_conf.get_stage_1_conf(cfg)


def get_stage_2_conf(cfg):
    return base_conf.get_stage_2_conf(cfg)


def get_stage_3_conf(cfg):
    return base_conf.get_stage_3_conf(cfg)
