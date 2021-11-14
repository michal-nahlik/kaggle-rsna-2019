import cv2
import pydicom
import numpy as np


def normal_window(data, window_center, window_width, intercept, slope):
    """
    Extract simple image window using given parameters.
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    :return: A single window extracted from given data
    """
    data = (data * slope + intercept)
    data_min = window_center - window_width // 2
    data_max = window_center + window_width // 2
    data[data < data_min] = data_min
    data[data > data_max] = data_max
    return data


def sigmoid_window(data, window_center, window_width, intercept, slope, U=1.0, eps=(1.0 / 255.0)):
    """
    Extract image window from data using given parameters and rescale values into sigmoid.
    https://www.kaggle.com/reppic/gradient-sigmoid-windowing
    :return: A single window extracted from given data where min value is 0 and max value is 1
    """
    data = (data * slope + intercept)
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * data + b
    data = U / (1 + np.power(np.e, -1.0 * z))
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


def get_first_of_dicom_field_as_int(x):
    """
    Get value x as int or extract first value from x if it is pydicom MultiValue
    """
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_window_metadata(data):
    """
    Get parameters needed for correct windowing from dicom metadata
    """
    intercept = data.RescaleIntercept
    slope = data.RescaleSlope
    window_center = get_first_of_dicom_field_as_int(data.WindowCenter)
    window_width = get_first_of_dicom_field_as_int(data.WindowWidth)

    return [window_center, window_width, intercept, slope]


def normalize(data):
    """
    Rescale data into values 0 - 255 and return it as int16
    """
    data = np.float32(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data *= 255
    return np.int16(data)


def load_and_preprocess(path, img_id, img_sz, window_type, window_conf):
    """
    Load data from dicom file and create image where RGB layers are used to store different window information. Final
    image is normalized to values 0-255, cast as int16 and resized to img_sz.
    :param path:
    :param img_id:
    :param img_sz:
    :param window_type: 0 - normal_window, 1-sigmoid window
    :param window_conf: 2d array (shape: (3,2)) with information about window center and width. -1 indicates that value
     should be load from dicom metadata. Example: [[-1, -1], [80, 200], [600, 2800]]
    :return: image as int16 3d array
    """
    data = pydicom.dcmread(path + img_id + '.dcm')
    window_center_metadata, window_width_metadata, intercept, slope = get_window_metadata(data)
    data = data.pixel_array

    img = np.zeros((*np.shape(data), len(window_conf)), dtype=np.int16)

    for i, window_data in enumerate(window_conf):
        window_center, window_width = window_data

        if window_width == -1:
            window_center = window_center_metadata
            window_width = window_width_metadata

        if window_type == 0:
            layer = normal_window(data, window_center, window_width, intercept, slope)
        else:
            layer = sigmoid_window(data, window_center, window_width, intercept, slope)

        img[..., i] = normalize(layer)

    if np.shape(img)[2] == 1:
        img = np.squeeze(img)
        img = np.repeat(img[..., None], 3, 2)

    if img_sz:
        img = cv2.resize(img, (img_sz, img_sz))

    return img


def load_dcm_data(path, img_id):
    """
    Load pixel array from dmc file with given image ID.
    """
    data = pydicom.dcmread(path + img_id + '.dcm')
    data = data.pixel_array
    return data


def load_and_preprocess_adj(path, img_id, img_sz, meta):
    """
    Load data from dicom file and create image where RGB layers are used to store current and adjacent slices. If there
    is no adjacent slice available the current slice is used instead. Final image contains full data (no window is used)
    and is not normalized (contains full information from dicom.pixel_array).
    :param path:
    :param img_id:
    :param img_sz:
    :param meta: metadata containing Image ID, PatientID, StudyInstanceUID ordered by ImagePositionPatient[2]
    :return: image as float32 3d array
    """
    data = load_dcm_data(path, img_id)
    img = np.zeros((*np.shape(data), 3), dtype=np.float32)

    img_meta = meta[meta['Image'] == img_id]
    patient_id = img_meta['PatientID'].values[0]
    sequence_id = img_meta['StudyInstanceUID'].values[0]
    seq_meta = meta[(meta['PatientID'] == patient_id) & (meta['StudyInstanceUID'] == sequence_id)]
    seq_meta.reset_index(drop=True, inplace=True)

    img_in_seg_meta = seq_meta[seq_meta['Image'] == img_id]
    img_in_seg_id = img_in_seg_meta.index.to_numpy()

    if img_in_seg_id == 0:
        img[..., 0] = data
    else:
        img[..., 0] = load_dcm_data(path, seq_meta.loc[img_in_seg_id - 1, 'Image'].values[0])

    img[..., 1] = data

    if img_in_seg_id == np.max(seq_meta.index.to_numpy()):
        img[..., 2] = data
    else:
        img[..., 2] = load_dcm_data(path, seq_meta.loc[img_in_seg_id + 1, 'Image'].values[0])

    if img_sz:
        img = cv2.resize(img, (img_sz, img_sz))

    return img
