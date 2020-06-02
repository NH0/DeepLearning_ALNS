from torch.utils.data import Dataset


class CVRPDataSet(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):

        return self.inputs[index], self.labels[index]

    def __len__(self):

        return len(self.labels)
