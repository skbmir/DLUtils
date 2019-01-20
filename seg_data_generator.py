import os
import random
import numpy as np
import cv2
from keras.utils import Sequence


# This vvvvv is for example_preprocess function and augs
# from albumentations import (
#     HorizontalFlip, VerticalFlip, Flip, Transpose, Rotate, ShiftScaleRotate, RandomScale,
#     RandomBrightness, RandomContrast, RandomBrightnessContrast, JpegCompression, Blur,
#     MedianBlur, Compose, OneOf
# )

class SegDataGenerator(Sequence):
    ''' Data generator class for segmentation

    Note:
        Used as data generator in fit_generator from keras.
        Includes support for augmentations via passing prepocessing function
        as preprocessing_function parameter. For example interface of preprocessing function
        see example_preprocess function.

    Args:
        input_directory (str): path to the folder where the input images are stored

        mask_directory (str): path to the folder where the masks are stored

        input_extention (str): extention of the input images files

        mask_extention (str): extention of the input masks files

        input_shape (tuple/list): target shape of the input images

        mask_shape (tuple/list): target shape of the masks

        batch_size (int): batch size

        preload_dataset (bool): if True input images and masks will be loaded to RAM (should be set to False if dataset if larger than available RAM)

        prob_aug (float): probability of getting augmented image

        preprocessing_function (func): function that performs preprocessing and augmentation (if needed) (see example_preprocess function)

    Attributes:
        no public attributes
    '''

    def __init__(self,
                 input_directory, mask_directory,
                 input_extention='.jpg', mask_extention='.png',
                 input_shape=(256, 256, 3), mask_shape=(256, 256, 1),
                 batch_size=4, preload_dataset=False, prob_aug=0.5,
                 preprocessing_function=None, classes_colors=None):

        self._dir = input_directory
        self._mask_dir = mask_directory
        self._in_shape = input_shape
        self._mask_shape = mask_shape
        self._fext = input_extention
        self._mext = mask_extention
        self._batch_size = batch_size

        in_files = list(filter(lambda x: x.endswith(self._fext), os.listdir(self._dir)))
        in_files.sort()
        mask_files = list(filter(lambda x: x.endswith(self._mext), os.listdir(self._mask_dir)))
        mask_files.sort()

        self._files = list()
        for i, name in enumerate(in_files):
            self._files.append((name, mask_files[i]))

        random.shuffle(self._files)

        self._preload = preload_dataset
        self._prob_aug = prob_aug
        self._data = None
        self._masks = None

        self._h = 0
        self._w = 1
        self._c = 2
        self._colors = classes_colors

        if (preprocessing_function is not None) and callable(preprocessing_function):
            self._preprocess = preprocessing_function
        else:
            self._preprocess = self._def_preprocess

        if self._preload:
            self._data = list()
            for i, names in enumerate(self._files):
                img = cv2.imread(os.path.join(self._dir, names[0]), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(os.path.join(self._mask_dir, names[1]), cv2.IMREAD_UNCHANGED)
                self._data.append((img, mask))

    def __len__(self):
        return int(np.ceil(len(self._files) / float(self._batch_size)))

    def __getitem__(self, idx):

        batch_x = np.empty(
            (self._batch_size, self._in_shape[self._h], self._in_shape[self._w], self._in_shape[self._c]),
            dtype='float32')
        batch_y = np.empty(
            (self._batch_size, self._mask_shape[self._h], self._mask_shape[self._w], self._mask_shape[self._c]),
            dtype='float32')

        if self._preload:

            for i, imgs in enumerate(self._data[idx * self._batch_size:(idx + 1) * self._batch_size]):
                batch_img, batch_mask = self.__filter__(imgs[0], imgs[1])
                batch_x[i] = batch_img
                batch_y[i] = batch_mask

        else:

            for i, names in enumerate(self._files[idx * self._batch_size:(idx + 1) * self._batch_size]):
                img = cv2.imread(os.path.join(self._dir, names[0]), cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(os.path.join(self._mask_dir, names[1]), cv2.IMREAD_UNCHANGED)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                batch_img, batch_mask = self.__filter__(img, mask)
                batch_x[i] = batch_img
                batch_y[i] = batch_mask

        return batch_x, batch_y

    def __filter__(self, img, mask):
        inter = cv2.INTER_AREA

        if (img.shape[self._w] < self._in_shape[self._w]) or (img.shape[self._h] < self._in_shape[self._h]):
            inter = cv2.INTER_CUBIC

        batch_img = cv2.resize(img, dsize=(self._in_shape[self._w], self._in_shape[self._h]), interpolation=inter)
        batch_mask = cv2.resize(mask, dsize=(self._mask_shape[self._w], self._mask_shape[self._h]),
                                interpolation=inter)

        batch_img_a, batch_mask_a = self._preprocess(batch_img, batch_mask, self._prob_aug)

        if self._mask_shape[self._c] > 1:
            dashed_mask = np.empty((self._mask_shape[self._h], self._mask_shape[self._w], self._mask_shape[self._c]),
                                   dtype='float32')

            for color in self._colors:
                one_mask = cv2.inRange(batch_mask_a, np.asarray(self._colors[color]), np.asarray(self._colors[color]))
                np.append(dashed_mask, one_mask.astype('float32'))
            return batch_img_a.astype('float32'), dashed_mask
        else:
            return batch_img_a.astype('float32'), batch_mask_a.astype('float32')

    @staticmethod
    def _def_preprocess(img, mask, prob_aug):
        ''' Default preprocessing and augmentation function for SegDataGenerator class

        Args:
            img (numpy.ndarray): input image as numpy array (loaded using opencv, skimage or other compatible modules)

            mask (numpy.ndarray): mask as numpy array (loaded using opencv, skimage or other compatible modules)

            prob_aug (float): probability of getting augmented image (if used)
    
        Returns:
            tuple: tuple of preprocessed (image, mask)
        '''
        return img, mask

# vvvvv Example augmentation and preprocessing function vvvvv Albumentation module must be installed
# def example_augs(p=0.5):
#     return Compose([
#         OneOf([
#             Flip(p=0.5),
#             Transpose(p=0.2),
#             Rotate(limit=90, interpolation=cv2.INTER_CUBIC, p=0.2),
#             ShiftScaleRotate(shift_limit=0.125,
#                              scale_limit=0.25,
#                              rotate_limit=90,
#                              interpolation=cv2.INTER_CUBIC, p=0.5),
#             RandomScale(scale_limit=0.2, interpolation=cv2.INTER_CUBIC, p=0.2)
#         ], p=0.75),
#         OneOf([
#             RandomBrightness(limit=0.1, p=0.5),
#             RandomContrast(limit=0.1, p=0.2),
#             RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1)
#         ], p=0.25),
#         JpegCompression(quality_lower=90, p=0.1),
#         OneOf([
#             Blur(blur_limit=3, p=0.1),
#             MedianBlur(blur_limit=5, p=0.1)
#         ], p=0.1)
#     ], p=p)

# def example_preprocess(img, mask, prob_aug):
#     ''' Example preprocessing and augmentation function for SegDataGenerator class

#     Args:
#         img (numpy.ndarray): input image as numpy array (loaded using opencv, skimage or other compatible modules)

#         mask (numpy.ndarray): mask as numpy array (loaded using opencv, skimage or other compatible modules)

#         prob_aug (float): probability of getting augmented image (if used)

#     Returns:
#         tuple: tuple of preprocessed (image, mask)
#     '''
#     augs = example_augs(p=prob_aug)
#     data = {'image': img, 'mask': mask}
#     augmented = augs(**data)
#     aimg = augmented['image']
#     amask = augmented['mask']
#     aimg_yuv = cv2.cvtColor(aimg, cv2.COLOR_BGR2YUV)
#     aimg_hls = cv2.cvtColor(aimg, cv2.COLOR_BGR2HLS)
#     clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(5,5))
#     yuv_split = cv2.split(aimg_yuv)
#     hls_split = cv2.split(aimg_hls)
#     yuv_split[0] = clahe.apply(yuv_split[0])
#     aimg = cv2.merge((yuv_split[0], hls_split[2], yuv_split[2]))
#     return aimg, amask
