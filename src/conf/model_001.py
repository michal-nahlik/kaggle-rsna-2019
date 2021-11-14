import conf.base_config_1 as base_conf


def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.model.source = 'efficientnet'
    cfg.model.name = 'efficientnet-b3'
    cfg.model.out_file_name = 'efficientnet-b3'
    return cfg


def get_stage_1_conf(cfg):
    return base_conf.get_stage_1_conf(cfg)


def get_stage_2_conf(cfg):
    return base_conf.get_stage_2_conf(cfg)


def get_stage_3_conf(cfg):
    return base_conf.get_stage_3_conf(cfg)
