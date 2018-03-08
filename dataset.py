# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random


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


class TwoStreamDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_segments=3,
                 new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_segments = num_segments

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        if not self.test_mode:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if self.modality == 'Flow':
                for item in tmp:
                    item[1] = int(item[1]) / 2
            tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))
        else:
            tmp = [x.strip().split() for x in open(self.list_file)]
            if self.modality == 'Flow':
                for item in tmp:
                    item[1] = int(item[1]) / 2
            # tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            # offsets = np.zeros((self.num_segments,))
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        if self.modality == 'Flow':
            offsets = offsets * 2 + 1
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        if self.modality == 'RGB':
            while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
                print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)) + ' not exists jumpping')
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        else:
            while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format('x', 1))):
                print(
                    os.path.join(self.root_path, record.path, self.image_tmpl.format('x', 1)) + ' not exists jumpping')
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]

        if not self.test_mode:
            sample_indice = [randint(low=1, high=record.num_frames + 2 - self.new_length)]
            if self.modality == 'Flow':
                sample_indice = sample_indice * 2 - 1  # flow index 1 3 5 7 ...
        else:
            sample_indice = self._get_val_indices(record)

        return self.get(record, sample_indice)

    def get(self, record, indice):

        images = list()

        for seg_ind in indice:
            p = int(seg_ind)
            for i in range(self.new_length):  # for optical flow getting a volumn start with seg_ind
                img = self._load_image(record.path, p)
                images.extend(img)
                if p < record.num_frames:
                    if self.modality == 'RGB':
                        p += 1
                    else:
                        p += 2

        # one image: H*W*C
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            # try:
            #     idx_skip = 1 + (idx-1)*5
            #     flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
            # except Exception:
            #     print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
            #     flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            # flow_x, flow_y, _ = flow.split()
            # x_img = flow_x.convert('L')
            # y_img = flow_y.convert('L')
            # image_tmpl='{:s}_{:05d}.jpg'  self.image_tmpl.format('x', 1) -> 'x_00001.jpg'
            x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        if not self.test_mode:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            for item in tmp:
                item[1] = int(item[1]) / 2
            tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))
        else:
            tmp = [x.strip().split() for x in open(self.list_file)]
            for item in tmp:
                item[1] = int(item[1]) / 2
            # tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:  # random sample
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:  # [0,0,1,1,1,2,2,3]     dense sample
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            # offsets = np.zeros((self.num_segments,))
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        return offsets + 1

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        if self.modality == 'RGB':
            while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
                print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)) + ' not exists jumpping')
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        else:
            while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format('x', 1))):
                print(
                    os.path.join(self.root_path, record.path, self.image_tmpl.format('x', 1)) + ' not exists jumpping')
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):  # for optical flow getting a volumn start with seg_ind
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # one image: H*W*C
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
