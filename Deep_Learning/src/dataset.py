import torch
from torch.utils.data import TensorDataset, DataLoader

class PyTorchDataset:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        # pandas -> numpy -> torch
        self.X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        self.y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

        self.X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        self.y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)

        self.X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        self.y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

        self.batch_size = batch_size

    def get_loaders(self):
        train_dataset = TensorDataset(self.X_train_t, self.y_train_t)
        val_dataset = TensorDataset(self.X_val_t, self.y_val_t)
        test_dataset = TensorDataset(self.X_test_t, self.y_test_t)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader