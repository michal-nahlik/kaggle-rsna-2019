import cv2
import torch
import numpy as np
import data.util as util
import data.dcm_util as dcm_util

from torch.utils.data import Dataset


class IntracranialDataset(Dataset):
    """
    General dataset that handles the non specific functions
    """

    def __init__(self, path, data, img_sz, transform=None, labels=True):

        self.full_data = data
        self.data = data
        self.labels = labels

        self.img_sz = img_sz
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'Image']

        img = self.get_data(img_name)
        img = util.center_image(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image_name': img_name, 'image': img, 'labels': labels}
        else:
            return {'image_name': img_name, 'image': img}

    def balance_data(self, seed):
        self.data = util.balance_data(self.full_data, seed)


class IntracranialDatasetPng(IntracranialDataset):
    """
    PNG data set loads data from png images saved on hdd.
    """

    def __init__(self, path, data, img_sz, transform=None, labels=False):
        super(IntracranialDatasetPng, self).__init__(path, data, img_sz, transform, labels)

    def get_data(self, img_name):
        return cv2.imread(self.path + img_name + '.png')


class IntracranialDatasetDcm(IntracranialDataset):
    """
    DCM dataset loads data from dicom files and creates image using window_type and window_conf. RGB layers are used
    to store different window information.
    """
    def __init__(self, path, data, img_sz, window_type, window_conf, transform=None, labels=False):
        super(IntracranialDatasetDcm, self).__init__(path, data, img_sz, transform, labels)
        self.window_type = window_type
        self.window_conf = window_conf

    def get_data(self, img_name):
        try:
            img = dcm_util.load_and_preprocess(self.path, img_name, self.img_sz, self.window_type, self.window_conf)
        except Exception as e:
            print(img_name, e)
            img = np.zeros((self.img_sz, self.img_sz, 3), dtype=np.float)

        return img


class IntracranialDatasetDcmAdj(IntracranialDataset):
    """
    ADJ dataset loads data from dicom files and create images where RGB layers are used to store current CT slice and
    adjacent slices without any windowing.
    """
    def __init__(self, path, data, meta, img_sz, transform=None, labels=False):
        super(IntracranialDatasetDcmAdj, self).__init__(path, data, img_sz, transform, labels)
        self.meta = meta

    def get_data(self, img_name):
        try:
            img = dcm_util.load_and_preprocess_adj(self.path, img_name, self.img_sz, self.meta)
        except Exception as e:
            print('Error: ', img_name, e)
            img = np.zeros((self.img_sz, self.img_sz, 3), dtype=np.float)

        return img