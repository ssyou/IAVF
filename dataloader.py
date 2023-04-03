import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import random


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    categories = ['negative', 'positive']
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    index = id_to_idx[ids]
    # y[index] = 1
    # for id in ids:
    #     index = id_to_idx[id]
    #     y[index] = 1
    return index



class LLP_dataset(Dataset):

    def __init__(self, label, dataset, audio_dir, video_dir, st_dir, transform=None, flag = "test", K=10, num_now = 0):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.dataset = dataset
        len_all = len(self.df)
        each_len_data = int(len_all/K)
        self.flag = flag
        if self.flag == "train":
            self.df_loc_now = self.df.loc[num_now * each_len_data: (num_now + 1) * each_len_data,]
            self.filenames = self.df["filename"][num_now * each_len_data: (num_now + 1) * each_len_data, ]
            self.start_point = num_now * each_len_data
        else:
            self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.flag == "train":
            idx = idx + self.start_point
            row = self.df_loc_now.loc[idx, :]
        else:
            row = self.df.loc[idx,:]
        if self.dataset == 'earthquake':
            row_split = row[0].split('_')
            audio_name = '{}_{}'.format(row_split[0], row_split[1])
            audio_label = int(row_split[2][0])
            visual_name = '{}_{}'.format(row_split[2][2:], row_split[3])
            visual_label = int(row_split[4][0])
        else:
            audio_name = row[0][:11]
            audio_label = int(row[0][12])
            visual_name = row[0][14:25]
            visual_label = int(int(row[0][26]))

        audio = np.load(os.path.join(self.audio_dir, audio_name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, visual_name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, visual_name + '.npy'))
        ids = row[0][-8:]
        label = ids_to_multinomial(ids)
        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 'total_label': label, 'audio_label': audio_label, 'visual_label': visual_label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        all_sample = []
        for idx in sample_idx:
            idx = idx + self.start_point
            row = self.df_loc_now.loc[idx, :]
            
            if self.dataset == 'earthquake':
                row_split = row[0].split('_')
                audio_name = '{}_{}'.format(row_split[0], row_split[1])
                audio_label = int(row_split[2][0])
                visual_name = '{}_{}'.format(row_split[2][2:], row_split[3])
                visual_label = int(row_split[4][0])
            else:
                audio_name = row[0][:11]
                audio_label = int(row[0][12])
                visual_name = row[0][14:25]
                visual_label = int(int(row[0][26]))

            audio = np.load(os.path.join(self.audio_dir, audio_name + '.npy'))
            video_s = np.load(os.path.join(self.video_dir, visual_name + '.npy'))
            video_st = np.load(os.path.join(self.st_dir, visual_name + '.npy'))
            ids = row[0][-8:]
            label = ids_to_multinomial(ids)
            sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 'total_label': label, 'audio_label': audio_label, 'visual_label': visual_label}
            all_sample.append(sample)
        return all_sample

class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']
            return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(label)}
        else:
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            total_label = sample['total_label']
            audio_label = sample['audio_label']
            visual_label = sample['visual_label']
            return {'audio': torch.from_numpy(audio), 'video_s': torch.from_numpy(video_s),
                    'video_st': torch.from_numpy(video_st),
                    'total_label': torch.tensor(total_label),
                    'audio_label': torch.tensor(audio_label),
                    'visual_label': torch.tensor(visual_label)}