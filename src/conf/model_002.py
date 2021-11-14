import conf.base_config_1 as base_conf


def get_default_conf():
    cfg = base_conf.get_default_conf()
    cfg.batch_size = 64
    cfg.model.source = 'torchhub'
    cfg.model.name = 'resnext101_32x8d_wsl'
    cfg.model.out_file_name = 'resnext101_32x8d_wsl'

    cfg.train.loader['batch_size'] = cfg.batch_size
    cfg.test.loader['batch_size'] = cfg.batch_size

    return cfg


def get_stage_1_conf(cfg):
    return base_conf.get_stage_1_conf(cfg)


def get_stage_2_conf(cfg):
    return base_conf.get_stage_2_conf(cfg)


def get_stage_3_conf(cfg):
    return base_conf.get_stage_3_conf(cfg)
