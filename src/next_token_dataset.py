import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# создание Dataset-ов
# Dataset для обучающей выборки: y смещен на 1 токен относительно x
class TrainDataset(Dataset):
    def __init__(self, indexed_texts):
        for text in indexed_texts:
            text.insert(len(text), 1)
        self.indexed_texts = indexed_texts
           
    def __len__(self):
        return len(self.indexed_texts)

    def __getitem__(self, idx):
        tokens = self.indexed_texts[idx]
        x = tokens[:-1]
        y = tokens[1:]
        return {
            'context_ids': torch.tensor(x, dtype=torch.long),
            'target_ids': torch.tensor(y, dtype=torch.long)
        }

# Dataset для валидационной выборки: x - 3/4 предложения, y - оставшаяся 1/4 предложения
class ValDataset(Dataset):
    def __init__(self, indexed_texts):
        self.indexed_texts = indexed_texts
           
    def __len__(self):
        return len(self.indexed_texts)

    def __getitem__(self, idx):
        tokens = self.indexed_texts[idx]
        split_point = int(len(tokens) // 4)
        x = tokens[:-split_point]
        y = tokens[-split_point:]
    
        return {
            'context_ids': torch.tensor(x, dtype=torch.long),
            'target_ids': torch.tensor(y, dtype=torch.long)
        }
    
# функция для пэддинга
def collate_fn(batch):
    context_ids = [item['context_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    context_ids = pad_sequence(context_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
    return {
        'context_ids': context_ids,
        'target_ids': target_ids
    }