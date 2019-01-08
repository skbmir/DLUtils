import os
import numpy as np
import cv2
from keras.utils import Sequence
from albumentations import (
    HorizontalFlip, VerticalFlip, Flip, Transpose, Rotate, ShiftScaleRotate, RandomScale,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, JpegCompression, Blur,
    MedianBlur, Compose, OneOf
)

class SegDataGenerator(Sequence):

    def __init__(self, 
                 input_directory, mask_directory, 
                 input_extention='.jpg', mask_extention='.png', 
                 input_shape=(256, 256, 3), mask_shape=(256, 256, 1), 
                 batch_size=4, preload_dataset=False, 
                 prob_aug=0.5, preprocessing_function=None):

        self._dir = input_directory
        self._mask_dir = mask_directory
        self._in_shape = input_shape
        self._mask_shape = mask_shape
        self._fext = input_extention
        self._mext = mask_extention
        self._batch_size = batch_size
        self._in_files = list(filter(lambda x: x.endswith(self._fext), os.listdir(self._dir)))
        self._in_files.sort()
        self._mask_files = list(filter(lambda x: x.endswith(self._mext), os.listdir(self._mask_dir)))
        self._mask_files.sort()
        self._preload = preload_dataset
        self._prob_aug = prob_aug
        self._data = None
        self._masks = None

        if (preprocessing_function is not None) and callable(preprocessing_function):
            self._preprocess = preprocessing_function
        else:
            self._preprocess = self._def_preprocess

        if self._preload:
            self._data = list()
            self._masks = list()
            for i, name in enumerate(self._files):
                img = cv2.imread(self._dir + name, cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(self._mask_dir + self._mask_files[i], cv2.IMREAD_UNCHANGED)
                self._data.append(img)
                self._masks.append(mask)

    def __len__(self):
        return int(np.ceil(len(self.files / float(self.batch_size))))
    
    def __getitem__(self, idx):
        h = 0
        w = 1
        c = 2

        batch_x = np.empty((self._batch_size, self._in_shape[h], self._in_shape[w], self._in_shape[c]), dtype='float32')
        batch_y = np.empty((self._batch_size, self._mask_shape[h], self._mask_shape[w], self._mask_shape[c]), dtype='float32')

        inter = cv2.INTER_AREA

        if self._preload:
            
            for i, img in enumerate(self._data[idx*self._batch_size:(idx+1)*self._batch_size]):

                if (img.shape[w] < self._in_shape[w]) or (img.shape[h] < self._in_shape[h]):
                    inter = cv2.INTER_CUBIC

                batch_img = cv2.resize(img, dsize=(self._in_shape[w], self._in_shape[h]), interpolation=inter)
                batch_mask = cv2.resize(self._masks[i], dsize=(self._mask_shape[w], self._mask_shape[h]), interpolation=inter)

                batch_img, batch_mask = self._preprocess(batch_img, batch_mask, self._prob_aug)
                batch_x[i] = batch_img
                batch_y[i] = batch_mask

        else:

            for i, name in enumerate(self._in_files[idx*self._batch_size:(idx+1)*self._batch_size]):

                batch_img = cv2.imread(self._dir + name, cv2.IMREAD_UNCHANGED)
                batch_mask = cv2.imread(self._mask_dir + self._mask_files, cv2.IMREAD_UNCHANGED)

                if (img.shape[w] < self._in_shape[w]) or (img.shape[h] < self._in_shape[h]):
                    inter = cv2.INTER_CUBIC
                
                batch_img = cv2.resize(batch_img, dsize=(self._in_shape[w], self._in_shape[h]), interpolation=inter)
                batch_mask = cv2.resize(batch_mask, dsize=(self._mask_shape[w], self._mask_shape[h]), interpolation=inter)

                batch_img, batch_mask = self._preprocess(batch_img, batch_mask, self._def_augs, self._prob_aug)
                batch_x[i] = batch_img
                batch_y[i] = batch_mask

        return batch_x, batch_y

    @staticmethod
    def _def_augs(p=0.5):
        return Compose([
            OneOf([
                Flip(p=0.5),
                Transpose(p=0.2),
                Rotate(limit=90, interpolation=cv2.INTER_CUBIC, p=0.2),
                ShiftScaleRotate(shift_limit=0.125,
                                 scale_limit=0.25,
                                 rotate_limit=90,
                                 interpolation=cv2.INTER_CUBIC, p=0.5),
                RandomScale(scale_limit=0.2, interpolation=cv2.INTER_CUBIC, p=0.2)
            ], p=0.75),
            OneOf([
                RandomBrightness(limit=0.1, p=0.5),
                RandomContrast(limit=0.1, p=0.2),
                RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1)
            ], p=0.25),
            JpegCompression(quality_lower=90, p=0.1),
            OneOf([
                Blur(blur_limit=3, p=0.1),
                MedianBlur(blur_limit=5, p=0.1)
            ], p=0.1)
        ], p=p)

    @staticmethod  
    def _def_preprocess(img, mask, aug_function, prob_aug):
        augs = aug_function(p=prob_aug)
        data = {'image': img, 'mask': mask}
        augmented = augs(**data)
        aimg = augmented['image']
        amask = augmented['mask']
        aimg_yuv = cv2.cvtColor(aimg, cv2.COLOR_BGR2YUV)
        aimg_hls = cv2.cvtColor(aimg, cv2.COLOR_BGR2HLS)
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(5,5))
        yuv_split = cv2.split(aimg_yuv)
        hls_split = cv2.split(aimg_hls)
        yuv_split[0] = clahe.apply(yuv_split[0])
        aimg = cv2.merge((yuv_split[0], hls_split[2], yuv_split[2]))
        return aimg, amask
