# Hotel Booking Cancellation Prediction

## 📌 Loyihaning maqsadi
Ushbu loyiha mehmonxonalarda buyurtmalarning bekor qilinish ehtimolini bashorat qilishga qaratilgan. Maqsad: **binary classification** modeli orqali `is_canceled` target ustuni bo‘yicha mehmonning buyurtmasi bekor qilinish ehtimolini aniqlash.

---
 ## 📌 Loyihaning xulqosaviy tahlili

Ushbu loyiha **Supervised Machine Learning (SML) – Binary Classification** turida amalga oshirilmoqda. Maqsad – mehmonning buyurtmasi bekor qilinish ehtimolini bashorat qilish (`is_canceled` target ustuni). Quyida loyiha bo‘yicha xulqosaviy fikrlar keltirilgan:

### 🔹 1. Maqsad va ahamiyat
- **Maqsad:** Mehmonxona menejerlari uchun buyurtma bekor qilinish ehtimolini oldindan aniqlash.  
- **Foyda:** Riskli buyurtmalarni oldindan aniqlash, resurslarni optimallashtirish va marketing strategiyasini yaxshilash.  

### 🔹 2. Supervised Machine Learning tanlovi
- Loyihada ishlatilgan SML modeli, chunki mavjud **tarixiy buyurtma ma’lumotlari** mavjud va target (`is_canceled`) aniq belgilangan.  
- Binary classification vazifasi, chunki target faqat **2 ta klassga** ega:  
  - 0 → Buyurtma bekor qilinmagan  
  - 1 → Buyurtma bekor qilingan  

### 3️⃣ `is_canceled` uchun eng muhim metrik

✅ **F1-Score** eng muhim, chunki:

- Dataset imbalanced bo‘lishi mumkin (bekor qilinadigan bookinglar kamroq).  
- Precision va Recall o‘rtasidagi balansni hisobga oladi.  
- Model nafaqat bekor qilinadigan bookinglarni topishi, balki noto‘g‘ri signal bermasligi kerak.  

**Masalan:**  
- Agar hotel har bir bekor qilishni 100% topishga harakat qilsa (Recall = 1), lekin noto‘g‘ri “bekor” deb aytsa (Precision past bo‘lsa), foyda kamayadi.  
- F1-Score bu ikki jihatni birlashtirib, eng real tavsiya beruvchi metrik bo‘ladi.


## 📊 Dataset haqida
Datasetda **32 ustun** mavjud. Asosiy ustunlar:  

---

### 🔹 3. Model tanlash va pipeline
- **Preprocessing:** missing values to‘ldirish, categorical features one-hot encoding, numeric features scaling  
- **Model tanlovi:** Logistic Regression, Random Forest, XGBoost kabi klassik binary classification modellari  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- **Feature importance:** qaysi ustunlar targetga eng ko‘p ta’sir qiladi 

### 🔹 4. Yakuniy xulosa
- Ushbu loyiha **SML – Binary Classification** konseptini real biznes ma’lumotlariga tatbiq qilishning yaxshi misoli hisoblanadi.  
- Model yordamida mehmonxona menejerlari **riskli buyurtmalarni oldindan aniqlash va resurslarni samarali boshqarish** imkoniyatiga ega bo‘ladi.  
- Loyihaning xulosalari va vizualizatsiyalari **qaror qabul qilish jarayonini optimallashtirish**ga xizmat qiladi.
- Riskli mehmonlarni aniqlash va marketing strategiyasini yaxshilash
- Peak months va shaharlarda resurslarni optimallashtirish




