import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        dataset = pd.read_csv(file)
        x = dataset.drop('label', axis=1).to_numpy()
        y = dataset['label'].to_numpy()

        standardScalar = MinMaxScaler()
        standardScalar.fit(x)
        x = standardScalar.transform(x)
        self.x = x.reshape(x.shape[0], 28, 28).astype('float32')
        self.label = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.transform(self.x[idx]), self.target_transform(self.label[idx])