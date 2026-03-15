from src.data_loader import DataLoaderClass
from src.dataset import PyTorchDataset
from src.model import StructuredNN
from src.train import Trainer

def main():

    data_loader = DataLoaderClass(
        "../data/X_train_selected_smote.csv",
        "../data/y_train_smote.csv",
        "../data/X_test_selected.csv",
        "../data/y_test.csv"
    )

    X_train, X_val, y_train, y_val, X_test, y_test = data_loader.load_data()

    dataset = PyTorchDataset(X_train, y_train, X_val, y_val, X_test, y_test)

    train_loader, val_loader, test_loader = dataset.get_loaders()

    input_dim = X_train.shape[1]

    model = StructuredNN(input_dim)

    trainer = Trainer(model, train_loader, val_loader)

    trainer.train()

if __name__ == "__main__":
    main()