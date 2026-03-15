import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, num_epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.num_epochs):
            # ===== TRAIN =====
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(self.train_loader.dataset)
            self.train_losses.append(train_loss)

            # ===== VALIDATION =====
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(self.val_loader.dataset)
            self.val_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def plot_losses(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()