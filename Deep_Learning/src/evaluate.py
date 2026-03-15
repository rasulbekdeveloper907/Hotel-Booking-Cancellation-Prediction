import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, X_test_t, y_test_t):
        self.model = model
        self.X_test_t = X_test_t
        self.y_test_t = y_test_t

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_test_t)
            probs = torch.sigmoid(logits)
            y_pred = (probs >= 0.5).int().cpu().numpy().flatten()
        y_true = self.y_test_t.cpu().numpy().flatten()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("\nTest Metrics")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Canceled", "Canceled"])
        plt.figure(figsize=(7,6))
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.show()