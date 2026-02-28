readme = """# Proyek Klasifikasi Gambar — GTSRB (German Traffic Sign)

## Ringkasan
Proyek ini membangun model **CNN (Sequential)** untuk klasifikasi rambu lalu lintas pada dataset **GTSRB**.
Hasil akhir model diekspor ke **SavedModel**, **TFLite**, dan **TFJS**.

---

## Dataset
- Sumber dataset: Kaggle — `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`
- Jumlah kelas: **43** (label **0–42**)
- Jumlah gambar:
  - Train folder: **39,209**
  - Test folder: **12,630**
  - Total: **51,839**
- Resolusi gambar: **tidak seragam** (bervariasi), dibuktikan dengan sampling ukuran gambar yang berbeda-beda.

File penting:
- `Train.csv` berisi label (`ClassId`) dan koordinat ROI (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`) serta path gambar.
- `Test.csv` berisi label test dan path gambar.

---

## Pembagian Data (Train/Validation/Test)
Langkah split yang dilakukan:
1. Menggabungkan seluruh metadata menjadi satu dataframe:
   - `df_all = concat(Train.csv, Test.csv)`
2. Melakukan pembagian **stratified** berdasarkan `ClassId` menjadi:
   - **Train 80%**
   - **Validation 10%**
   - **Test 10%**

---

## Preprocessing (tanpa preprocessing permanen di file asli)
Preprocessing dilakukan saat loading dataset:
1. Load image dari path pada CSV
2. **Crop ROI** menggunakan koordinat ROI dari CSV
3. Resize ke **128×128**
4. Konversi ke `float32` (range 0–255) lalu normalisasi di dalam model dengan `Rescaling(1./255)`

---

## Arsitektur Model (Sequential + Conv2D + Pooling)
Model CNN menggunakan `tf.keras.Sequential` dengan komponen utama:
- `Rescaling(1./255)`
- data augmentation ringan (rotation/zoom/translation)
- Blok Conv + Pool:
  - `Conv2D(32) + MaxPooling2D`
  - `Conv2D(64) + MaxPooling2D`
  - `Conv2D(128) + MaxPooling2D`
  - `Conv2D(256) + MaxPooling2D`
- `GlobalAveragePooling2D`
- `Dense(256) + Dropout`
- Output: `Dense(43, softmax)`

---

## Callback yang digunakan
- `ModelCheckpoint` (menyimpan model terbaik berdasarkan `val_accuracy`)
- `EarlyStopping` (menghindari overfitting)
- `ReduceLROnPlateau` (menurunkan learning rate saat stagnan)

---

## Hasil
- Akurasi training (best): ~**0.998**
- Akurasi testing (best): ~**0.998**  
Memenuhi syarat minimal akurasi training dan testing **≥ 85%**.

Notebook juga menampilkan plot:
- `accuracy` vs `val_accuracy`
- `loss` vs `val_loss`

---

## Export Model (SavedModel, TFLite, TFJS)
Model inference dibuat **tanpa layer augmentation** untuk kompatibilitas ekspor.

Output berada pada folder `submission/`:
- `saved_model/` : format **SavedModel**
- `tflite/model.tflite` + `tflite/label.txt` : format **TensorFlow Lite**
- `tfjs_model/model.json` + shard `.bin` : format **TensorFlow.js**

---

## Bukti Inference
Inference dilakukan pada 1 sampel dari test set dengan hasil:
- Prediksi **SavedModel/Keras**: benar
- Prediksi **TFLite**: benar dan konsisten dengan SavedModel

Contoh output inference pada notebook:
- `True label: 16`
- `Pred label: 16`
- `Top prob: 1.0`