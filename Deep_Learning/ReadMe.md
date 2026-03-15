# Hotel Booking Cancellation Prediction

## 📂 Dataset manbasi

- **X_train_selected_smote.csv** – SMOTE yordamida balanslangan trening features.
- **y_train_smote.csv** – Trening labels (0 = Not Canceled, 1 = Canceled).
- **X_test_selected.csv** – Test features (feature selection qo‘llangan).
- **y_test.csv** – Test labels.

## 🔹 Train/Validation/Test Split

- `train_test_split` yordamida SMOTE bilan olingan trening ma’lumotlarning **10%** validation uchun ajratilgan.
- **Stratifikatsiya** orqali target label balansini saqlash: `stratify=y_train`.

## 🏗 Model Arxitekturasi (StructuredNN)

- **PyTorch** `nn.Module` yordamida yaratilgan.
- **Tarmoq qatlamlari**:
  1. Input layer → 32 neurons → ReLU → Dropout(0.2)
  2. Hidden layer → 16 neurons → ReLU
  3. Output layer → 1 neuron (sigmoid orqali binary classification)
- **Activation**:
  - Hidden layerlarda `ReLU`
  - Output layer uchun `BCEWithLogitsLoss` ishlatilgan → sigmoid bilan birga

## ⚙️ Training konfiguratsiyasi

- **Loss Function**: `BCEWithLogitsLoss` (binary classification uchun)
- **Optimizer**: Adam, learning rate = 0.001
- **Epochlar**: 50
- **Training loop**:
  - Har epochda:
    - `train_loss` va `val_loss` hisoblangan
    - `optimizer.zero_grad() → loss.backward() → optimizer.step()`
  - Validation step `torch.no_grad()` bilan amalga oshirilgan
- **Input features**: `X_train_t.shape[1]` bilan aniqlanadi (feature selection qo‘llangan)
- **Output**: 1 neuron (0 yoki 1)
- **Batch Size**: 32
- **Dropout**: 0.2 → overfittingni kamaytirish uchun

## 📊 Baholash metrikalari

- Accuracy
- Precision
- Recall
- F1-score

### Confusion Matrix
- Vizualizatsiya `ConfusionMatrixDisplay` yordamida amalga oshirilgan.

## 🛠 Texnologiyalar

| Komponent          | Texnologiya / Kutubxona | Tavsif                                               |
| ------------------ | ----------------------- | ---------------------------------------------------- |
| Data Handling      | pandas, numpy           | CSV o‘qish, tensor konvertatsiyasi                   |
| Data Preprocessing | scikit-learn            | SMOTE bilan balanslash, train/validation split       |
| Model              | PyTorch (nn.Module)     | Structured Feedforward Neural Network, Dropout bilan |
| Loss               | BCEWithLogitsLoss       | Binary Classification uchun                          |
| Optimizer          | Adam                    | Gradient descent asosida optimizatsiya               |
| Metrics            | sklearn.metrics         | Accuracy, Precision, Recall, F1, Confusion Matrix    |
| Visualization      | matplotlib              | Learning Curve, Confusion Matrix                     |

---