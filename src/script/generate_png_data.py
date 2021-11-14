import cv2
import data.factory as data_factory
import utils.config as config
import data.util as data_util
import data.dcm_util as dcm_util

import os
from tqdm import tqdm
from joblib import Parallel, delayed

cfg = config.Config()
cfg.path = '../data/'

cfg.train = dict()
cfg.train.folder = 'stage_2_train_images/'
cfg.train.df_name = 'stage_2_train.csv'
cfg.train.meta = 'stage_2_train_metadata.csv'
cfg.train.path = cfg.path + cfg.train.folder

cfg.test = dict()
cfg.test.folder = 'stage_2_test_images/'
cfg.test.df_name = 'stage_2_sample_submission.csv'
cfg.test.path = cfg.path + cfg.test.folder


# Load data frames
train_df, _ = data_factory.load_train_dataframes(cfg)
test_df = data_util.load_csv(cfg.path + cfg.test.df_name)


def save_img(folder, name, img):
    cv2.imwrite(folder + name + '.png', img)


def prep_and_save(path, target, row, img_sz):
    try:
        img = dcm_util.load_and_preprocess(path, row['Image'], img_sz, window_type, window_conf)
        save_img(target, row['Image'], img)
    except Exception as e:
        print(row['Image'], e)


path_train = cfg.train.path
path_test = cfg.test.path

# Normal window data 256
window_conf = [[-1, -1], [80, 200], [600, 2800]]
window_type = 0  # 0 - normal, 1 - sigmoid
img_sz = 256

path_train_out = cfg.path + 'stage_1_train_images_{}_png/'.format(img_sz)
os.mkdir(path_train_out)
Parallel(n_jobs=-1)(delayed(prep_and_save)(path_train, path_train_out, row, img_sz) for index, row in tqdm(train_df.iterrows()))

path_test_out = cfg.path + 'stage_1_test_images_{}_png/'.format(img_sz)
os.mkdir(path_test_out)
Parallel(n_jobs=-1)(delayed(prep_and_save)(path_test, path_test_out, row, img_sz) for index, row in tqdm(test_df.iterrows()))

# Sigmoid window data 378
window_conf = [[-1, -1], [80, 200], [600, 2800]]
window_type = 1  # 0 - normal, 1 - sigmoid
img_sz = 378

path_train_out = cfg.path + 'stage_1_train_sig_images_{}_png/'.format(img_sz)
os.mkdir(path_train_out)
Parallel(n_jobs=-1)(delayed(prep_and_save)(path_train, path_train_out, row, img_sz) for index, row in tqdm(train_df.iterrows()))

path_test_out = cfg.path + 'stage_1_test_sig_images_{}_png/'.format(img_sz)
os.mkdir(path_test_out)
Parallel(n_jobs=-1)(delayed(prep_and_save)(path_test, path_test_out, row, img_sz) for index, row in tqdm(test_df.iterrows()))
