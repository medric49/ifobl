import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from PIL import Image
from torch.utils.data.dataset import T_co

import utils


class VideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, episode_len, cam_ids, to_lab=False, im_w=64, im_h=64):
        self._root = Path(root)
        self._num_classes = len(list(self._root.iterdir()))

        self.im_w = im_w
        self.im_h = im_h
        self.update_files()

        self._episode_len = episode_len + 1
        self._cam_ids = cam_ids
        self.to_lab = to_lab

    def update_files(self, max_num_video=None):
        self._files = []
        for c in range(self._num_classes):
            class_dir = self._root / str(c)
            files = list(sorted(class_dir.iterdir()))
            if max_num_video is not None and len(files) > max_num_video:
                old_files = files[:-max_num_video]
                files = files[-max_num_video:]
                for f in old_files:
                    os.remove(f)
            self._files.append(files)

    def _sample(self):
        if len(self._cam_ids) > 1:
            cam1, cam2 = random.sample(self._cam_ids, k=3)
        else:
            cam1, cam2 = 0, 0

        classes = list(range(self._num_classes))
        class_1 = 0
        classes.remove(class_1)
        class_2 = random.choice(classes)

        video_i = random.choice(self._files[class_1])
        video_n = random.choice(self._files[class_2])

        video_i = np.load(video_i)[cam1, :self._episode_len]
        video_n = np.load(video_n)[cam2, :self._episode_len]

        if tuple(video_i.shape[1:3]) != (self.im_h, self.im_w):
            video_i = VideoDataset.resize(video_i, self.im_w, self.im_h)
        if tuple(video_n.shape[1:3]) != (self.im_h, self.im_w):
            video_n = VideoDataset.resize(video_n, self.im_w, self.im_h)

        if self.to_lab:
            video_i = VideoDataset.rgb_to_lab(video_i)
            video_n = VideoDataset.rgb_to_lab(video_n)

        video_i = video_i.transpose(0, 3, 1, 2).copy()
        video_n = video_n.transpose(0, 3, 1, 2).copy()

        return video_i, video_n

    @staticmethod
    def resize(video, im_w, im_h):
        frame_list = []
        for t in range(video.shape[0]):
            frame = Image.fromarray(video[t])
            frame = np.array(frame.resize((im_w, im_h), Image.BICUBIC), dtype=np.float32)
            frame_list.append(frame)
        frame_list = np.stack(frame_list)
        return frame_list

    @staticmethod
    def rgb_to_lab(video):
        T = video.shape[0]
        return np.array([utils.rgb_to_lab(video[t]) for t in range(T)], dtype=np.float32)

    @staticmethod
    def sample_from_dir(video_dir, episode_len=None):
        if episode_len is not None:
            episode_len += 1
        else:
            episode_len = -1

        video_dir = Path(video_dir)
        files = list(video_dir.iterdir())
        video_i = np.load(random.choice(files))[0, :episode_len]
        return video_i

    @staticmethod
    def transform_frames(frames, im_w, im_h, to_lab):
        if tuple(frames.shape[1:3]) != (im_h, im_w):
            frames = VideoDataset.resize(frames, im_w, im_h)
        if to_lab:
            frames = VideoDataset.rgb_to_lab(frames)
        return frames

    @staticmethod
    def augment(video_i: torch.Tensor, video_n: torch.Tensor):
        T = video_i.shape[1]
        p_list = [0.05 for i in range(T)]
        indices = [i for i in range(T) if np.random.rand() > p_list[i]]
        video_i = video_i[:, indices, :, :, :]
        video_n = video_n[:, indices, :, :, :]
        return video_i, video_n

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
