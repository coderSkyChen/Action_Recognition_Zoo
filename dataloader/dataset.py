# @Time    : 2018/10/18 11:07
# @File    : dataset.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn
import torch.utils.data as data
import random
import math
import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        if len(self._data) == 2:
            return -1
        else:
            return int(self._data[2])


class DataSet(data.Dataset):
    def __init__(self, root_path, list_file, modality, image_tmpl, transform):
        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _get_train_indices(self, record):
        pass

    def _get_test_indices(self, record):
        pass

    def __getitem__(self, index):
        pass

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)+1
            nl = 1 if self.modality == 'RGB' else 5
            for i in range(nl):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class TSNDataSet(DataSet):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, test_mode=False):
        DataSet.__init__(self, root_path, list_file, modality, image_tmpl, transform)

        self.num_segments = num_segments
        self.new_length = new_length
        self.new_length = 5
        self.test_mode = test_mode
        self._parse_list()

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._get_train_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

