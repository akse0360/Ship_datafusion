import torch
from torch.utils.data import Dataset

import random
class ais_dataset(Dataset):
    def __init__(self, data):
        self.data = data # list of ais tracks grouped by mmsi
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        track = [list(range(4)) for _ in range(100)]
        t = random.randint(0, len(track)-2)
        
        ais1 = track[t]
        ais2 = track[t+1]

        return ais1, ais2
    