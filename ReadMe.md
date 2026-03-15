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



# 🔧 Technical Contribution
## Hotel Booking Cancellation Prediction

Ushbu loyihada mehmonxona bron qilish bekor qilinishini oldindan bashorat qilish uchun
to‘liq **Machine Learning pipeline** ishlab chiqildi va u **baseline modellardan
ensemble yondashuvlargacha** bosqichma-bosqich takomillashtirildi.

---

## 📊 1. Baseline Model Evaluation

Dastlab muammoni tushunish va taqqoslash (benchmark) yaratish maqsadida quyidagi
klassik Machine Learning algoritmlaridan foydalanildi:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7837 | 0.7004 | 0.7415 | 0.7204 | 0.8705 |
| Random Forest | 0.8827 | 0.8718 | 0.8064 | 0.8378 | 0.9514 |
| Decision Tree | 0.8413 | 0.7850 | 0.7955 | 0.7902 | 0.8323 |
| K-Nearest Neighbors | 0.7355 | 0.6238 | 0.7461 | 0.6795 | 0.8155 |

**Xulosa:**  
Random Forest eng yaxshi baseline natijani ko‘rsatdi, biroq modelning
umumlashuvchanligini oshirish uchun yanada kuchli yondashuv zarur edi.

---

## 🚀 2. Ensemble Algorithms orqali Modelni Rivojlantirish

Baseline natijalarni yaxshilash maqsadida bir nechta **ensemble learning**
algoritmlari joriy etildi:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| **Stacking Classifier** | **0.8815** | **0.8568** | **0.8220** | **0.8391** | **0.9482** |
| Bagging (Random Forest) | 0.8798 | 0.8677 | 0.8023 | 0.8337 | 0.9507 |
| Voting Classifier | 0.8736 | 0.8425 | 0.8163 | 0.8292 | 0.9426 |
| Gradient Boosting | 0.8502 | 0.8111 | 0.7836 | 0.7971 | 0.9288 |
| AdaBoost | 0.8273 | 0.7528 | 0.8045 | 0.7778 | 0.9090 |

**Asosiy natija:**  
Stacking Classifier eng muvozanatli natijani ko‘rsatdi va F1-score hamda Recall
ko‘rsatkichlari sezilarli darajada yaxshilandi.

---

## 🧹 3. Data Leakage’ni Oldini Olish

Modelning real sharoitda to‘g‘ri ishlashini ta’minlash uchun target bilan kuchli
bog‘liq yoki voqeadan keyingi ma’lumotlarni o‘z ichiga olgan featurelar olib tashlandi:

- ❌ `reservation_status`
- ❌ `reservation_status_date`  
  *(faqat undan hosila bo‘lgan sana featurelar saqlandi)*

Bu qadam **data leakage** muammosini oldini olishda muhim ahamiyatga ega bo‘ldi.

---

## 📅 4. Date Feature Engineering

`reservation_status_date` ustunidan quyidagi yangi vaqtga oid featurelar yaratildi:

- `res_year`
- `res_month`
- `res_day`
- `res_weekday`

Qo‘shimcha ravishda:
- `arrival_date_month` (string) → `arrival_month_num` (numeric)

**Natija:**  
Model mavsumiylik va vaqtga bog‘liq naqshlarni aniqroq o‘rganishga erishdi.

---

## ➕ 5. Aggregated & Ratio Feature Engineering

Domain knowledge asosida yuqori informatsiyali yangi featurelar yaratildi:

- `total_stay_nights`  
  (`stays_in_weekend_nights + stays_in_week_nights`)

- `total_guests`  
  (`adults + children + babies`)

- `adr_per_person`  
  (`adr / total_guests`)

- `special_req_ratio`  
  (`total_of_special_requests / total_stay_nights`)

Bu featurelar modelga oddiy ustunlarga nisbatan kuchliroq signal berdi.

---

## 🚩 6. Binary / Flag Featurelar

Quyidagi 0/1 ko‘rinishidagi mantiqiy featurelar yaratildi:

- `has_children`
- `is_long_stay` (≥ 7 tun)
- `has_parking`
- `has_deposit`
- `changed_room`  
  *(reserved va assigned room mos kelmasligi)*

---

## 🧠 7. Rare Category Handling (Memory-Safe)

One-Hot Encoding jarayonida xotira sarfini kamaytirish va overfitting’ni oldini olish
uchun kam uchraydigan kategoriyalar `"Other"` guruhiga birlashtirildi.

Qo‘llanilgan ustunlar:
- `country`
- `agent`
- `company`
- `city`

---

## ✅ Yakuniy Xulosa

- Baseline modellardan **ensemble yondashuvlarga** o‘tish orqali model sifati oshirildi  
- Feature engineering orqali real biznes mantiqi modelga singdirildi  
- Data leakage to‘liq bartaraf etildi  
- Ishlab chiqarishga tayyor (**production-ready**) ML pipeline yaratildi


# 💼 Business Contribution
## Hotel Booking Cancellation Prediction

Ushbu loyiha mehmonxona biznesida eng muhim muammolardan biri bo‘lgan
**bronlarni bekor qilinishini (cancellation)** oldindan bashorat qilish orqali
**daromadni oshirish, xarajatlarni kamaytirish va operatsion qarorlarni
yaxshilashga** qaratilgan.

---

## 🎯 1. Business Muammoni Aniqlash

Mehmonxonalar uchun booking cancellation quyidagi salbiy oqibatlarga olib keladi:
- ❌ Xonalar bo‘sh qolishi (revenue loss)
- ❌ Overbooking va noto‘g‘ri rejalashtirish
- ❌ Marketing va operatsion resurslarning samarasiz sarfi

**Loyiha maqsadi:**  
Yuqori ehtimollik bilan bekor qilinadigan bronlarni **oldindan aniqlash** va
mehmonxonaga **proaktiv qaror qabul qilish imkonini berish**.

---

## 💰 2. Daromadni Optimallashtirish (Revenue Optimization)

Model yordamida:
- Bekor bo‘lish ehtimoli yuqori bo‘lgan bronlar aniqlanadi
- Xonalarni **qayta sotish (re-selling)** imkoniyati oshadi
- Dinamik narxlash (dynamic pricing) strategiyalarini qo‘llash mumkin bo‘ladi

**Business impact:**
- Bo‘sh qolgan xonalar soni kamayadi
- Umumiy bandlik darajasi (occupancy rate) oshadi

---

## 🧠 3. Risk-Based Decision Making

Model chiqishlari asosida mehmonxona:
- Yuqori riskli mijozlar uchun:
  - Oldindan to‘lov (prepayment)
  - Depozit siyosatini kuchaytirish
- Past riskli mijozlar uchun:
  - Moslashuvchan bekor qilish shartlari

**Natija:**  
Riskka asoslangan adolatli va samarali biznes siyosati.

---

## 📣 4. Marketing Strategiyalarini Takomillashtirish

Model natijalari marketing bo‘limiga quyidagicha yordam beradi:
- Bekor qilish ehtimoli yuqori bo‘lgan mijozlarga:
  - Reminder xatlar
  - Chegirmalar
  - Maxsus takliflar
- Past riskli segmentlarga:
  - Upsell / cross-sell kampaniyalari

**Natija:**  
Marketing xarajatlari kamayadi, konversiya darajasi oshadi.

---

## 🏨 5. Operatsion Rejalashtirishni Yaxshilash

Cancellation bashoratlari asosida:
- Housekeeping rejalari optimallashtiriladi
- Xodimlar smenasi aniqroq belgilanadi
- Oziq-ovqat va boshqa resurslar ortiqcha xarid qilinmaydi

**Business impact:**
- Operatsion xarajatlar kamayadi
- Xizmat sifati barqarorlashadi

---

## 📊 6. Ma’lumotga Asoslangan Boshqaruv (Data-Driven Management)

Loyiha orqali:
- Intuitsiyaga emas, **real ma’lumotlarga asoslangan qarorlar** qabul qilinadi
- Rahbariyat uchun:
  - Cancellation risk dashboard
  - Segmentlar bo‘yicha tahlillar

yaratish imkoniyati paydo bo‘ladi.

---

## ⚖️ 7. Mijoz Tajribasini Yaxshilash

Risk darajasiga qarab:
- Halol va moslashuvchan siyosat
- Keraksiz cheklovlarsiz mijozlarga qulay shartlar

**Natija:**  
- Mijoz ishonchi oshadi  
- Brendga sodiqlik kuchayadi

---

## 📈 Yakuniy Business Qiymat

- 💵 Daromad yo‘qotilishi kamaydi
- 🧩 Resurslardan samarali foydalanildi
- 🎯 Marketing va pricing strategiyalari aniqroq bo‘ldi
- 🧠 Data-driven madaniyat shakllandi

---

**Xulosa:**  
Ushbu loyiha Machine Learning modelini **real biznes qiymatiga aylantirdi** va
mehmonxona uchun strategik ustunlik yaratdi.




