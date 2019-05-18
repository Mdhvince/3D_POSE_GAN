import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class MPii2DPoseDataset(Dataset):

    def __init__(self, json_file, transform=None):
    	self.df = pd.read_json(json_file)
    	self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
    	joints = np.array(self.df.loc[idx, 'joints']).astype('float')
    	joints = joints.reshape(-1, 2)

    	if self.transform:
    		joints = self.transform(joints)

    	return joints







