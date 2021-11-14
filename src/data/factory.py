import pandas as pd
import data.util as util
import data.rsna_dataset as rsna_data
import post.make_submission as sub

from torch.utils.data import DataLoader


def get_data_loader(cfg, data_cfg, data_df, meta_df=None):
    """
    Create data loader using given config and data.
    :param cfg: expected values: cfg.use_png, cfg.window_type, cfg.img_sz, cfg.window_conf
    :param data_cfg: expected values: data_cfg.path, data_cfg.transform, data_cfg.labels
    :param data_df: DataFrame containing transformed data (see util.transform_df)
    :param meta_df: used in case of sequence dataset to extract information about image Z position
    :return: DataLoader
    """

    if cfg.use_png:
        dataset = rsna_data.IntracranialDatasetPng(data=data_df, path=data_cfg.path, img_sz=cfg.img_sz,
                                                   transform=data_cfg.transform, labels=data_cfg.labels)
    else:
        if cfg.window_type is None:
            dataset = rsna_data.IntracranialDatasetDcmAdj(data=data_df,  meta=meta_df, path=data_cfg.path, img_sz=cfg.img_sz,
                                                         transform=data_cfg.transform, labels=data_cfg.labels)

        else:
            dataset = rsna_data.IntracranialDatasetDcm(data=data_df, path=data_cfg.path, img_sz=cfg.img_sz,
                                                       transform=data_cfg.transform, labels=data_cfg.labels,
                                                       window_type=cfg.window_type, window_conf=cfg.window_conf)

    return DataLoader(dataset, **data_cfg.loader)


def get_fold_data_loaders(cfg, data_df, patient_df, meta_df=None):
    """
    Create train and validation data loader using given config and patient data to for split
    :param cfg: expected values: cfg.n_folds, cfg.fold, cfg.seed, cfg.train
    :param data_df: DataFrame containing transformed data (see util.transform_df)
    :param patient_df: DataFrame with label on patient level used to split data without leak in train/val set
    :param meta_df: used in case of sequence dataset to extract information about image Z position
    :return: DataLoader: data_loader_train, DataLoader: data_loader_val
    """
    train_df, val_df = util.get_fold_data(data_df, patient_df, cfg.n_folds, cfg.fold, cfg.seed)

    data_loader_train = get_data_loader(cfg, cfg.train, train_df, meta_df)
    data_loader_val = get_data_loader(cfg, cfg.train, val_df, meta_df)

    return data_loader_train, data_loader_val


def load_train_dataframes(cfg):
    """
    Load and transform training data, training meta data and creates a patient level labels data frame
    :param cfg: expected values: cfg.path, cfg.train.df_name, cfg.train.meta
    :return: data_df, patient_df, data_meta_df
    """
    data_df = util.load_csv(cfg.path + cfg.train.df_name)
    data_meta_df = pd.read_csv(cfg.path + cfg.train.meta)
    data_meta_df = data_meta_df.rename(columns={"ID": "Image"})
    data_df = pd.merge(data_df, data_meta_df[['Image', 'PatientID']], on='Image')
    patient_df = data_df.groupby(['PatientID']).any()
    patient_df.head()

    # Drop bad data
    data_df = data_df.drop(data_df.loc[data_df['Image'] == 'ID_6431af929'].index, axis=0).reset_index(drop=True)
    data_df = data_df.drop(data_df.loc[data_df['Image'] == 'ID_0077bc852'].index, axis=0).reset_index(drop=True)

    return data_df, patient_df, data_meta_df


def get_pseudo_labels(submission_list, low_thr, high_thr):
    """
    Create data frame with pseudo labels based on submissions in submission_list given prediction thresholds. Only
    images where all classes satisfy the thresholds are used.
    :param submission_list: list of submission names in output_path/submissions/ that should be used to create pseudo labels
    :param low_thr: threshold for 0 label
    :param high_thr: threshold for 1 label
    :return: DataFrame: pseudo_labels
    """
    comb, _ = sub.ensemble(submission_list)
    comb = util.transform_df(comb)
    prob = comb.values[:, 1:]
    prob_high = (prob < low_thr) + (prob > high_thr)
    prob_high = prob_high.all(axis=1)

    labels = comb.loc[prob_high]
    labels.reset_index(drop=True)

    pseudo_labels = (labels.iloc[:, 1:] > 0.5) * 1
    pseudo_labels.insert(0, 'Image', labels['Image'])
    pseudo_labels.reset_index(drop=True)

    return pseudo_labels
