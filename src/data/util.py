import cv2
import numpy as np
import pandas as pd

from scipy import ndimage
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler


def load_csv(csv_path):
    """
    Load csv on given path and transform it using #transform_df.
    """
    data_df = pd.read_csv(csv_path)
    return transform_df(data_df)


def transform_df(data_df):
    """
    Transform pandas dataframe so each image id has only one row with column for each class.
    """
    data_df[['ID', 'Image', 'Diagnosis']] = data_df['ID'].str.split('_', expand=True)
    data_df = data_df.drop(['ID'], axis=1).drop_duplicates()
    data_df = data_df.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
    data_df['Image'] = 'ID_' + data_df['Image']
    return data_df


def center_image(img):
    """
    Center image based on center of mass of its pixel values.
    """
    if np.sum(img) == 0:
        return img

    img_sz = np.shape(img)

    center_of_mass = ndimage.measurements.center_of_mass(img)
    shift_y = (img_sz[0] / 2) - center_of_mass[0]
    shift_x = (img_sz[1] / 2) - center_of_mass[1]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    dst = cv2.warpAffine(img, M, (img_sz[0], img_sz[1]))
    return dst


def get_fold_data(data_df, patient_df, n_folds, fold, seed):
    """
    Split data using stratified k-fold on 'any' column on patient level so there is no patient overlap in training
    and validation subset.
    """
    skf = StratifiedKFold(n_splits=n_folds, random_state=seed)
    train_index, test_index = list(skf.split(patient_df, patient_df['any']))[fold]
    patient_train = patient_df.iloc[train_index]
    patient_test = patient_df.iloc[test_index]

    fold_train_df = data_df.loc[data_df['PatientID'].isin(patient_train.index.values)]
    fold_val_df = data_df.loc[data_df['PatientID'].isin(patient_test.index.values)]

    return fold_train_df.reset_index(drop=True), fold_val_df.reset_index(drop=True)


def balance_data(data_df, rus_seed):
    """
    Resample data using RandomUnderSampler on 'any' column.
    """
    rus = RandomUnderSampler(random_state=rus_seed)
    data_res, y_res = rus.fit_resample(data_df, data_df[['any']])
    data_res_df = data_df[data_df[['Image']].isin(data_res[:, 0]).values]

    return data_res_df.reset_index(drop=True)


def print_stats(data_df):
    """
    Print information about the diagnoses distribution in given data frame and return the summary as data frame.
    """
    y = data_df.groupby(['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']).ngroup()

    print('Length {}'.format(len(data_df)))
    print('Empty {}'.format((data_df[['any']] == 0).sum().values))

    sums = data_df[['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']].sum()
    prct = data_df[['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']].mean()
    stats = pd.concat([sums, prct], keys=['Sums', 'Percent'], axis=1)

    return stats
