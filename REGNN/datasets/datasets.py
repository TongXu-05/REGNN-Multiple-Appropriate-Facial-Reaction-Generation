import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset



class ActionData(Dataset):
    def __init__(self, root, data_type, neighbors, split_dir, augmentation=None,
                 num_frames=50, stride=50, neighbor_pattern='nearest'):
        self.root = os.path.join(root, 'ActionData')
        with open(split_dir) as js:
            split_data = json.load(js)[data_type]

        self.neighbors = neighbors
        self.aug = augmentation
        self.num_frames = num_frames
        self.stride = stride
        self.neighbor_pattern = neighbor_pattern
        self.all_data = self.get_site_id_clip(split_data, data_type)

    def get_site_id_clip(self, split_data, data_type):
        all_data = []
        for s in split_data:
            site, group, clip = s.split('/')
            all_data.extend([[data_type, site, group, 'speaker', clip, str(i)]
                             for i in range(0, 750 - self.num_frames + 1, self.stride)])

            all_data.extend([[data_type, site, group, 'listener', clip, str(i)]
                             for i in range(0, 750 - self.num_frames + 1, self.stride)])

        return all_data

    def __getitem__(self, index):
        dtype_site_group_pid_clip_idx = self.all_data[index]

        v_inputs = self.load_video_pth(dtype_site_group_pid_clip_idx)
        a_inputs = self.load_audio_pth(dtype_site_group_pid_clip_idx)
        # a_inputs = 0.0

        if self.neighbor_pattern in {'nearest', 'all'}:
            targets = '+'.join(dtype_site_group_pid_clip_idx)
        else:
            pair_site_group_pid_clip_idx = dtype_site_group_pid_clip_idx[:]
            pid = pair_site_group_pid_clip_idx[2]
            pair_site_group_pid_clip_idx[2] = {'speaker': 'listener', 'listener': 'speaker'}[pid]
            targets = self.load_npy(pair_site_group_pid_clip_idx).transpose(1, 0)

        return v_inputs, a_inputs, targets

    def __len__(self):
        return len(self.all_data)

    def load_video_pth(self, dtype_site_group_pid_clip_idx):
        dtype, site, group, pid, clip, idx = dtype_site_group_pid_clip_idx
        idx = int(idx)
        video_pth = os.path.join(self.root, dtype, 'video_pth', site, group, pid, clip + '.pth')
        video_inputs = torch.load(video_pth, map_location='cpu')[idx:idx + self.num_frames]
        return video_inputs

    def load_audio_pth(self, dtype_site_group_pid_clip_idx):
        dtype, site, group, pid, clip, idx = dtype_site_group_pid_clip_idx
        idx = int(idx)
        audio_pth = os.path.join(self.root, dtype, 'audio_pth', site, group, pid, clip + '.pth')
        audio_inputs = torch.load(audio_pth, map_location='cpu')[idx:idx + self.num_frames]
        return audio_inputs

    def load_npy(self, dtype_site_group_pid_clip_idx):
        if len(dtype_site_group_pid_clip_idx) == 6:
            dtype, site, group, pid, clip, idx = dtype_site_group_pid_clip_idx
        else:
            dtype, site, group, pid, clip = dtype_site_group_pid_clip_idx
            idx = -1

        idx = int(idx)
        pth = os.path.join(self.root, '*', 'npy', site, group, pid, clip + '.npy')

        npy_pth = glob.glob(pth)[0]
        npy_inputs = np.load(npy_pth)[idx: idx + self.num_frames] if not idx == -1 else np.load(npy_pth)
        return torch.from_numpy(npy_inputs).float()
