import pandas as pd

from torch.utils.data import Dataset
from pathlib2 import Path

class EncDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        self.data = pd.read_pickle(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        data = self.data.iloc[index]
        tokens_1 = self.tokenizer.tokenize(data['effect_sentence_1'], max_len=self.max_len)
        tokens_2 = self.tokenizer.tokenize(data['effect_sentence_2'], max_len=self.max_len)
        return tokens_1, tokens_2, data['same_action'], data['same_object_feature']
        

    def __len__(self):
        return len(self.data)