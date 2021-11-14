import torch
import utils.config as config

from albumentations.pytorch import ToTensor
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)


transform_train = Compose([
    OneOf([
        HorizontalFlip(0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.075, rotate_limit=25, p=0.5),
        GridDistortion(num_steps=20, distort_limit=0.25, interpolation=1, border_mode=4, p=0.5),
        ElasticTransform(alpha=1, sigma=15, alpha_affine=5, p=0.5),
        ], p=0.75),
    ToTensor()
])

transform_test= Compose([
    ToTensor()
])

transform_test_tta = Compose([
    HorizontalFlip(1),
    ToTensor()
])


def get_default_conf():

    cfg = config.Config()
    cfg.path = '../data/'
    cfg.output_path = '../output/'
    cfg.seed = 201910

    cfg.use_png = True
    cfg.fold = 0
    cfg.n_folds = 5
    cfg.n_classes = 6
    cfg.img_sz = 378
    cfg.batch_size = 64
    cfg.num_workers = 4

    cfg.train = dict()
    cfg.train.folder = 'stage_2_train_sig_images_378_png/'
    cfg.train.df_name = 'stage_2_train.csv'
    cfg.train.meta = 'stage_2_train_metadata.csv'
    cfg.train.path = cfg.path + cfg.train.folder
    cfg.train.transform = transform_train
    cfg.train.labels = True

    cfg.train.loader = dict(
                shuffle=True,
                batch_size=cfg.batch_size,
                drop_last=True,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )

    cfg.test = dict()
    cfg.test.folder = 'stage_2_test_sig_images_378_png/'
    cfg.test.df_name = 'stage_2_sample_submission.csv'
    cfg.test.path = cfg.path + cfg.test.folder
    cfg.test.transform = None
    cfg.test.transform_list = [transform_test, transform_test_tta]
    cfg.test.labels = False

    cfg.test.loader=dict(
                shuffle=False,
                batch_size=cfg.batch_size,
                drop_last=False,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )

    cfg.model = dict()
    cfg.model.device = torch.device("cuda:0")
    cfg.model.use_amp = True
    cfg.model.n_epochs = 4
    cfg.model.source = 'efficientnet'
    cfg.model.name = 'efficientnet-b0'
    cfg.model.out_file_name = 'efficientnet-b0'
    cfg.model.init_lr = 1e-3
    cfg.model.balance_data = False
    cfg.model.rus_seed = 2019
    cfg.model.scheduler = None

    cfg.model.optim = dict(
        name='RAdam',
        params=dict(
            lr=cfg.model.init_lr,
        )
    )

    cfg.model.loss = dict(
        name='BCEWithLogitsLoss',
        params=dict(),
    )

    return cfg

def get_stage_1_conf(cfg):
    cfg.model.stage = 1
    return cfg

def get_stage_2_conf(cfg):
    cfg.model.init_lr = 1e-5
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

def get_stage_3_conf(cfg):
    cfg.model.stage = 3
    cfg.model.n_epochs = 2
    cfg.model.init_lr = 8e-6
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
            gamma=1 / 2,
        )
    )

    return cfg
