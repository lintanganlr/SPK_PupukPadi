# SPK Penentuan Dosis Pupuk Padi — PT Petrokimia Gresik (v2)

Sistem Pendukung Keputusan berbasis web yang memberikan rekomendasi dosis pupuk
(Petroganik Premium, Urea, SP-36, K-Plus) untuk tanaman padi sawah berdasarkan
kondisi tanah, faktor lingkungan, dan target panen.

## 🆕 Apa yang Berubah di v2?

| Aspek | v1 (lama) | v2 (sekarang) |
|---|---|---|
| **Dataset** | `Dataset_Skripsi_Lintang_Final.xlsx` | `dataset_ujitanah.xlsx` |
| **Pickle model** | Hanya 1 file `model.pkl` (model "terbaik") | **2 file: `model_rf.pkl` + `model_xgb.pkl`** |
| **Pemilihan algoritma** | Otomatis pakai model terbaik | **User memilih sendiri lewat dropdown di kalkulator** |
| **Hasil prediksi** | Selalu pakai 1 algoritma | User bisa langsung bandingkan RF vs XGBoost |
| **Endpoint `/predict`** | Tidak terima parameter algoritma | Terima parameter `algorithm` (`rf` / `xgb`) |

User sekarang bisa **memilih sendiri** algoritma yang ingin dipakai (Random Forest atau XGBoost)
sebelum menekan tombol Hitung, dan tersedia tombol "Hitung Ulang dengan Algoritma Lain" untuk
membandingkan kedua hasil dengan satu klik tanpa harus mengisi ulang form.

---

## Struktur Folder

```
SPK_Pupuk_Padi/
├── train_model.py                # Script training (jalankan SEKALI)
├── app.py                        # Flask web server (load 2 model)
├── requirements.txt              # Daftar library Python
├── dataset_ujitanah.xlsx         # <-- DATASET BARU
│
├── model/                        # (otomatis terbuat oleh train_model.py)
│   ├── model_rf.pkl              # Random Forest Regressor
│   ├── model_xgb.pkl             # XGBoost Regressor
│   ├── label_encoder.pkl         # Encoder Jenis_Tanah
│   └── metadata.pkl              # Info kedua model + evaluasi
│
├── templates/                    # File HTML
│   ├── layout.html
│   ├── home.html
│   ├── kalkulator.html           # Sekarang ada DROPDOWN algoritma
│   ├── edukasi.html
│   └── tentang.html              # Tampilkan evaluasi KEDUA model
│
└── static/                       # Asset web
    ├── css/style.css
    ├── js/main.js                # Kirim parameter `algorithm` ke /predict
    └── images/
        ├── Pupuk Petroganik Premium.png
        ├── Pupuk Urea.png
        ├── PupukSp-36.png
        └── Pupuk K Plus.png
```

---

## Cara Menjalankan (3 langkah)

### Langkah 1 — Install library

Buka terminal/PowerShell di folder `SPK_Pupuk_Padi`, lalu jalankan:

```bash
pip install -r requirements.txt
```

### Langkah 2 — Latih kedua model (jalankan SEKALI saja)

Pastikan file `dataset_ujitanah.xlsx` sudah ada di folder ini, lalu:

```bash
python train_model.py
```

Output yang akan muncul kira-kira seperti ini:

```
=================================================================
  TRAINING MODEL - SPK PENENTUAN DOSIS PUPUK PADI (v2)
  Menyimpan KEDUA model: Random Forest & XGBoost
=================================================================

Dataset dimuat: 3352 baris x 13 kolom

[1/6] Preprocessing data...
  Data bersih: 28xx baris (dihapus ~5xx)
  Kelas Jenis_Tanah: {'berpasir': 0, 'lempung': 1, 'liat': 2}

[2/6] Split data: 22xx latih | 5xx uji

[3/6] Melatih Random Forest Regressor...
  RF  -> RMSE: xx.xxxx | MAE: xx.xxxx | R2: 0.xxxx

[4/6] Melatih XGBoost Regressor...
  XGB -> RMSE: xx.xxxx | MAE: xx.xxxx | R2: 0.xxxx

[5/6] Perbandingan model (sistem skor 3 metrik):
  Skor RF  : x/3
  Skor XGB : x/3
  TERBAIK  : XGBoost Regressor (atau Random Forest)
  Catatan  : Kedua model TETAP DISIMPAN

[6/6] Menyimpan kedua model...

=================================================================
KEDUA MODEL BERHASIL DISIMPAN!
=================================================================
  -> model/model_rf.pkl       (Random Forest Regressor)
  -> model/model_xgb.pkl      (XGBoost Regressor)
  -> model/label_encoder.pkl  (3 kelas tanah)
  -> model/metadata.pkl       (info kedua model)
```

Setelah selesai, akan muncul folder `model/` berisi 4 file `.pkl`.

### Langkah 3 — Jalankan web

```bash
python app.py
```

Output:

```
==============================================================
  KEDUA MODEL BERHASIL DIMUAT
  - Random Forest : Random Forest Regressor
  - XGBoost       : XGBoost Regressor
  - Model terbaik : XGBoost Regressor (informatif)
==============================================================
 * Running on http://127.0.0.1:5000
```

Buka browser ke **http://127.0.0.1:5000**, lalu klik menu **Kalkulator**.

---

## Alur Sistem (versi 2)

```
[User Input di Kalkulator]
   pH, C-Organik, N, P, K, Curah Hujan, Suhu, Jenis Tanah, Target Panen
   + PILIHAN ALGORITMA (dropdown: Random Forest atau XGBoost)
        |
        v
[Frontend (kalkulator.html + main.js)]
   Validasi -> POST JSON ke /predict
   payload sekarang ada field `algorithm`: "rf" atau "xgb"
        |
        v
[Backend (app.py)]
   1. Pilih model sesuai algorithm di payload (MODELS[algo_key])
   2. Encode jenis_tanah pakai LabelEncoder dari training
   3. Susun fitur dengan urutan SAMA seperti training
   4. model.predict(features)
   5. Clip nilai negatif -> 0
        |
        v
[Response JSON]
   {
     petroganik_premium, urea, sp36, kplus,
     algoritma_key ("rf" / "xgb"),
     nama_model ("Random Forest Regressor" / "XGBoost Regressor"),
     info_aplikasi
   }
        |
        v
[Frontend menampilkan]
   - Banner: "dihitung dengan algoritma <nama>"
   - 4 kartu dosis (kg/ha)
   - Tombol "Hitung Ulang dengan Algoritma Lain"
     (memakai input yang sama, tinggal 1 klik)
   - Tabel cara & waktu pengaplikasian
```

---

## Catatan Penting

1. **Urutan fitur HARUS sama** antara training dan prediksi. File `metadata.pkl`
   menyimpan urutan ini, dan `app.py` memaksa urutan kolom dengan
   `features_df[feature_cols]` agar tidak ada kesalahan.

2. **LabelEncoder yang sama** harus dipakai. Jika dataset Anda di-update dan
   muncul jenis tanah baru, jalankan ulang `python train_model.py` agar
   encoder ikut ter-update.

3. **Jenis tanah yang tidak dikenal** akan otomatis di-fallback ke kelas
   pertama, dan response akan menyertakan field `warning`.

4. **Endpoint `/predict` menerima parameter `algorithm`** dengan nilai:
   - `"rf"` atau `"random_forest"` → Random Forest Regressor
   - `"xgb"` atau `"xgboost"` → XGBoost Regressor
   - Default (kalau tidak dikirim): `"rf"`

5. Untuk men-deploy ke production, ganti `debug=True` menjadi `debug=False`
   pada baris terakhir `app.py`, dan gunakan WSGI server seperti Gunicorn
   atau Waitress (bukan `app.run()` Flask).

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError: dataset_ujitanah.xlsx` | Letakkan file dataset di folder yang sama dengan `train_model.py` |
| `Model belum tersedia` di web | Jalankan dulu `python train_model.py` |
| `model_rf.pkl` / `model_xgb.pkl` tidak ditemukan | Hapus folder `model/` dan jalankan ulang `train_model.py` |
| Dropdown jenis tanah kosong | Model belum dilatih → jalankan `train_model.py` |
| Hasil prediksi negatif | Sudah otomatis di-clip ke 0 di `app.py` |
| Port 5000 sudah dipakai | Ubah `port=5000` di akhir `app.py` ke port lain (mis. 5001) |
| Algoritma yang dipilih tidak dikenal | Pastikan dropdown bernilai `rf` atau `xgb` |

---

## Contoh Pemakaian API `/predict`

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 5.85,
    "c_organik": 1.79,
    "n_total": 0.14,
    "p_total": 173,
    "k_total": 47.75,
    "curah_hujan": 158,
    "suhu": 27.3,
    "jenis_tanah": "lempung",
    "target_panen": 5,
    "algorithm": "xgb"
  }'
```

Response:

```json
{
  "petroganik_premium": 850.25,
  "urea": 273.10,
  "sp36": 124.80,
  "kplus": 168.40,
  "algoritma_key": "xgb",
  "nama_model": "XGBoost Regressor",
  "info_aplikasi": { ... }
}
```
