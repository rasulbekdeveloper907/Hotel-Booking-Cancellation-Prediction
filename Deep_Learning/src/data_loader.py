import pandas as pd

from sklearn.model_selection import train_test_split

class DataLoaderClass:
    def __init__(self, X_train_path, y_train_path, X_test_path, y_test_path):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path

    def load_data(self, test_size=0.1, random_state=42):
        # CSV o'qish
        X_train = pd.read_csv(self.X_train_path)
        y_train = pd.read_csv(self.y_train_path)
        X_test = pd.read_csv(self.X_test_path)
        y_test = pd.read_csv(self.y_test_path)

        # Train/Validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size,
            random_state=random_state, stratify=y_train
        )

        return X_train, X_val, y_train, y_val, X_test, y_test