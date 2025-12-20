import pandas as pd
import logging
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import joblib
import plotly.graph_objects as go

# Source papkasini qo'shish
source_path = os.path.abspath("../Source")
if source_path not in sys.path:
    sys.path.append(source_path)
from preprocessing import Cleaner, Encoder, Scaler

# Logging sozlamalari
log_path = r"C:\Users\Rasulbek907\Desktop\Hotel Booking Cancellation Prediction\Log\data_loader.log"
logging.basicConfig(filename=log_path, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# CSV faylni o'qish
csv_path = r"C:\Users\Rasulbek907\Desktop\Hotel Booking Cancellation Prediction\Data\Raw_Data\hotel_bookings_updated_2024.csv"
df = pd.read_csv(csv_path)
logging.info(f"Fayl o'qildi: {len(df)} satr, {len(df.columns)} ustun")

# Target va features
y = df['is_canceled']
X = df.drop(columns=['is_canceled'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing
cleaner = Cleaner(); cleaner.fit(X_train)
X_train_clean = cleaner.transform(X_train)
X_test_clean = cleaner.transform(X_test)

encoder = Encoder(max_unique=5); encoder.fit(X_train_clean)
X_train_enc = encoder.transform(X_train_clean)
X_test_enc = encoder.transform(X_test_clean)

scaler = Scaler(); scaler.fit(X_train_enc)
X_train_final = scaler.transform(X_train_enc)
X_test_final = scaler.transform(X_test_enc)

# SMOTE bilan balanslash
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_final, y_train)

# Base va ensemble classifiers
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42)

bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
vote = VotingClassifier(estimators=[('lr', lr), ('dt', dt), ('rf', rf), ('knn', knn)], voting='soft')
stack = StackingClassifier(estimators=[('lr', lr), ('dt', dt), ('rf', rf)], final_estimator=GradientBoostingClassifier(), cv=5)

models = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf, "KNN": knn, 
          "Gradient Boosting": gb, "HistGradientBoosting": hgb, "Bagging": bag, "Voting": vote, "Stacking": stack}

# Model training va metriklar
results = []
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_final)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred),4),
        "Precision": round(precision_score(y_test, y_pred),4),
        "Recall": round(recall_score(y_test, y_pred),4),
        "F1-Score": round(f1_score(y_test, y_pred),4)
    })

results_df = pd.DataFrame(results)
print(results_df)

# F1 bo'yicha eng yaxshi model va saqlash
best_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model = models[best_model_name]

save_dir = r"C:\Users\Rasulbek907\Desktop\Hotel Booking Cancellation Prediction\Models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{best_model_name.replace(' ','_')}_model.pkl")
joblib.dump(best_model, save_path)

print(f"Eng yaxshi model: {best_model_name}")
print(f"Model saqlandi: {save_path}")

# Plotly interaktiv jadval
colors = []
for i, row in results_df.iterrows():
    row_colors = []
    for metric in ['Accuracy','Precision','Recall','F1-Score']:
        if row[metric] >= 0.8:
            row_colors.append('lightgreen')
        elif row[metric] < 0.6:
            row_colors.append('lightcoral')
        else:
            row_colors.append('white')
    colors.append(['white'] + row_colors)

fig = go.Figure(data=[go.Table(
    header=dict(values=list(results_df.columns), fill_color='paleturquoise', align='center'),
    cells=dict(values=[results_df[col] for col in results_df.columns], fill_color=colors, align='center'))
])
fig.show()
