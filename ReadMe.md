# 🏨 Hotel Booking Cancellation Prediction

## 📌 Loyihaning Maqsadi
Ushbu loyiha mehmonxonalarda buyurtmalarning bekor qilinish ehtimolini bashorat qilishga qaratilgan.  
Maqsad: **binary classification** modeli orqali `is_canceled` target ustuni bo‘yicha mehmonning buyurtmasi bekor qilinish ehtimolini aniqlash.

---

## 📊 Loyihaning Xulqosaviy Tahlili

Loyiha ikki turdagi yondashuvni o‘rganadi:  
1. **Supervised Machine Learning (SML)** – Random Forest, Logistic Regression va boshqa klassik algoritmlar  
2. **Deep Learning (DL)** – PyTorch bilan yaratilgan Structured Feedforward Neural Network

### 🔹 Maqsad va Ahamiyat
- **Maqsad:** Mehmonxona menejerlari uchun buyurtma bekor qilinish ehtimolini oldindan aniqlash  
- **Foyda:** Riskli buyurtmalarni aniqlash, resurslarni optimallashtirish va marketing strategiyasini yaxshilash  

### 🔹 Binary Classification
- Target: `is_canceled`  
- Klasslar:  
  - 0 → Buyurtma bekor qilinmagan  
  - 1 → Buyurtma bekor qilingan  

---

## 📊 Dataset haqida
- 32 ustun mavjud  
- Asosiy ustunlar: mehmon xonasi, booking vaqti, mijoz turi, maxsus talablar va boshqalar  

---

## 🔹 Model Tanlash va Pipeline

### 🔹 SML (Supervised Machine Learning)
- **RandomForestClassifier** (Ensemble Tree, Bagging)  
- Binary classification uchun  
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Feature importance bilan eng ta’sirli ustunlar aniqlangan  

### 🔹 DL (Deep Learning)
- **Structured Neural Network** (PyTorch)  
- Input → 32 neurons → ReLU → Dropout(0.2) → 16 neurons → ReLU → Output(1 neuron)  
- **Loss:** BCEWithLogitsLoss  
- **Optimizer:** Adam  
- **Epoch:** 50  
- **Batch size:** 32  
- **Metrics:** Accuracy, Precision, Recall, F1-Score  

---

## 📊 DL vs SML Model Natijalari

| Algorithm                     | Model Type                  | Accuracy | Precision | Recall  | F1 Score | ROC-AUC |
|-------------------------------|----------------------------|---------|-----------|--------|----------|---------|
| 🔹 Structured Neural Network  | Deep Learning (DL)          | 0.8382  | 0.8068    | 0.7485 | 0.7766   | -       |
| 🌳 RandomForestClassifier     | Supervised ML (SML)         | 0.8795  | 0.8672    | 0.8021 | 0.8334   | 0.9513  |

**Izoh:**  
- 🔹 **DL modeli:** feedforward neural network, Dropout bilan overfitting kamaytirildi  
- 🌳 **SML modeli:** RandomForestClassifier eng yaxshi baseline SML natijani ko‘rsatdi  

---

## 🔹 Feature Engineering va Preprocessing
1. Missing values to‘ldirildi  
2. Categorical features → One-Hot Encoding  
3. Numeric features → StandardScaler  
4. Date features, aggregated va ratio features yaratildi  
5. Rare categories → "Other" guruhiga birlashtirildi  
6. Binary / flag features yaratildi: `has_children`, `is_long_stay`, `has_parking`, `has_deposit`, `changed_room`  

---

## 🔹 Data Leakage’ni Oldini Olish
- ❌ `reservation_status` va `reservation_status_date` olib tashlandi  
- Faqat hosila bo‘lgan vaqt featurelar saqlandi  

---

## 💼 Business Contribution

### 1. 💰 Daromadni optimallashtirish
- Riskli buyurtmalar aniqlanadi  
- Xonalarni qayta sotish (reselling) imkoniyati oshadi  
- Dynamic pricing strategiyasini qo‘llash mumkin  

### 2. 🧠 Risk-Based Decision Making
- Yuqori riskli mijozlar: oldindan to‘lov, kuchaytirilgan deposit siyosati  
- Past riskli mijozlar: moslashuvchan bekor qilish shartlari  

### 3. 📣 Marketing strategiyasi
- Reminder, chegirmalar va maxsus takliflar yuborish  
- Upsell / cross-sell kampaniyalar  

### 4. 🏗 Operatsion rejalashtirish
- Housekeeping va xodim smenalari optimallashtirildi  
- Oziq-ovqat va resurslar ortiqcha xarid qilinmaydi  

### 5. 📊 Data-Driven Management
- Qarorlar real ma’lumotlarga asoslangan  
- Dashboard va segmentlar bo‘yicha tahlil  

### 6. 🌟 Mijoz tajribasi
- Halol va moslashuvchan siyosat  
- Mijoz ishonchi va brendga sodiqlik oshadi  

---

## ✅ Yakuniy Xulosa
- Baseline SML modellardan Deep Learning va ensemble yondashuvlar orqali sifat oshirildi  
- Feature Engineering real biznes logikasini modelga kiritdi  
- 🔹 DL modeli: feedforward neural network, Dropout bilan overfitting kamaytirildi  
- 🌳 SML modeli: RandomForestClassifier, eng yaxshi ROC-AUC va F1 natija berdi  
- Loyiha **production-ready ML pipeline** sifatida ishga tayyor  
- Business qiymat: daromadni oshirish, xarajatlarni kamaytirish, marketing va operatsion qarorlarni optimallashtirish  