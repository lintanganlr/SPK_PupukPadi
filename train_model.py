"""
============================================================
  TRAINING SCRIPT - SPK PENENTUAN DOSIS PUPUK PADI (v3)
============================================================
Script ini dijalankan SEKALI untuk:
  1. Melatih model Random Forest & XGBoost
  2. Mengevaluasi & membandingkan kedua algoritma
  3. Menyimpan MODEL RANDOM FOREST ke folder model/
     (Random Forest digunakan sebagai model SPK)
  4. Menyimpan LabelEncoder untuk Jenis_Tanah
  5. Menyimpan metadata (evaluasi kedua model untuk
     halaman Tentang, fitur, kelas tanah)

Output:
  model/model_rf.pkl       -> Random Forest Regressor (model SPK)
  model/label_encoder.pkl  -> encoder Jenis_Tanah
  model/metadata.pkl       -> info evaluasi + fitur

Cara menjalankan:
  python train_model.py

Setelah selesai, jalankan: python app.py
============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 65)
print("  TRAINING MODEL - SPK PENENTUAN DOSIS PUPUK PADI (v3)")
print("  Model aktif: Random Forest Regressor")
print("=" * 65)

FILE_PATH = "dataset_ujitanah.xlsx"
if not os.path.isfile(FILE_PATH):
    FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_ujitanah.xlsx"
    )

if not os.path.isfile(FILE_PATH):
    raise FileNotFoundError(
        "File 'dataset_ujitanah.xlsx' tidak ditemukan. "
        "Pastikan file dataset berada di folder yang sama dengan train_model.py"
    )

df = pd.read_excel(FILE_PATH)
print(f"\nDataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")


# ============================================================
# 2. PREPROCESSING (sesuai proposal Section 3.3.4)
# ============================================================
print("\n[1/6] Preprocessing data...")

# Standarisasi nama kolom
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace('.', '_')
    .str.replace('-', '_')
)

# Whitespace normalization pada Jenis_Tanah
df['Jenis_Tanah'] = (
    df['Jenis_Tanah']
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        '0'   : 'tidak_diketahui',
        ''    : 'tidak_diketahui',
        'nan' : 'tidak_diketahui',
    })
)

# Label Encoding
le = LabelEncoder()
df['Jenis_Tanah_Encoded'] = le.fit_transform(df['Jenis_Tanah'])

# Konversi numerik
NUMERICAL_COLS = [
    'pH', 'C_Organik', 'N_Total', 'P_Total', 'K_Total',
    'Target_Panen', 'Curah_Hujan', 'Suhu',
    'P_Organik_Prem', 'Urea', 'SP36', 'Kplus',
]
df[NUMERICAL_COLS] = df[NUMERICAL_COLS].apply(pd.to_numeric, errors='coerce')

# Hapus missing value & duplikat
sebelum = len(df)
df = df.dropna(subset=NUMERICAL_COLS + ['Jenis_Tanah_Encoded'])
df = df.drop_duplicates().reset_index(drop=True)
print(f"  Data bersih: {len(df)} baris (dihapus {sebelum - len(df)})")
print(f"  Kelas Jenis_Tanah: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# ============================================================
# 3. SPLIT FITUR & TARGET
# ============================================================
FEATURE_COLS = [
    'pH', 'C_Organik', 'N_Total', 'P_Total', 'K_Total',
    'Target_Panen', 'Curah_Hujan', 'Suhu', 'Jenis_Tanah_Encoded',
]
TARGET_COLS = ['P_Organik_Prem', 'Urea', 'SP36', 'Kplus']

X = df[FEATURE_COLS].copy()
y = df[TARGET_COLS].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[2/6] Split data: {X_train.shape[0]} latih | {X_test.shape[0]} uji")


# ============================================================
# 4. FUNGSI EVALUASI MODEL (rata-rata 4 target + per target)
# ============================================================
def evaluasi(y_true, y_pred, target_cols):
    """Mengembalikan dict: rata-rata + detail per target pupuk."""
    rmse_list, mae_list = [], []
    per_target = {}

    for i, col in enumerate(target_cols):
        rmse = float(np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])))
        mae  = float(mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]))
        r2   = float(r2_score(y_true.iloc[:, i], y_pred[:, i]))
        per_target[col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        rmse_list.append(rmse)
        mae_list.append(mae)

    return {
        'RMSE'        : float(np.mean(rmse_list)),
        'MAE'         : float(np.mean(mae_list)),
        'R2'          : float(r2_score(y_true, y_pred)),
        'per_target'  : per_target,
    }


# ============================================================
# 5. MELATIH KEDUA MODEL
# ============================================================
print("\n[3/6] Melatih Random Forest Regressor...")
rf_base = RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt',
    random_state=42, n_jobs=-1,
)
model_rf = MultiOutputRegressor(rf_base)
model_rf.fit(X_train, y_train)
eval_rf = evaluasi(y_test, model_rf.predict(X_test), TARGET_COLS)
print(f"  RF  -> RMSE: {eval_rf['RMSE']:.4f} | MAE: {eval_rf['MAE']:.4f} | R2: {eval_rf['R2']:.4f}")

print("\n[4/6] Melatih XGBoost Regressor...")
xgb_base = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0,
)
model_xgb = MultiOutputRegressor(xgb_base)
model_xgb.fit(X_train, y_train)
eval_xgb = evaluasi(y_test, model_xgb.predict(X_test), TARGET_COLS)
print(f"  XGB -> RMSE: {eval_xgb['RMSE']:.4f} | MAE: {eval_xgb['MAE']:.4f} | R2: {eval_xgb['R2']:.4f}")


# ============================================================
# 6. PERBANDINGAN MODEL (informatif — disimpan di metadata)
#    Sistem skor 3 metrik: MAE rendah, RMSE rendah, R2 tinggi
#    Random Forest digunakan sebagai model SPK.
# ============================================================
skor_rf  = (int(eval_rf['RMSE'] < eval_xgb['RMSE']) +
            int(eval_rf['MAE']  < eval_xgb['MAE'])  +
            int(eval_rf['R2']   > eval_xgb['R2']))
skor_xgb = 3 - skor_rf

if skor_rf > skor_xgb:
    nama_terbaik = "Random Forest Regressor"
elif skor_xgb > skor_rf:
    nama_terbaik = "XGBoost Regressor"
else:
    nama_terbaik = ("Random Forest Regressor"
                    if eval_rf['R2'] >= eval_xgb['R2']
                    else "XGBoost Regressor")

print(f"\n[5/6] Perbandingan model (sistem skor 3 metrik):")
print(f"  Skor RF  : {skor_rf}/3")
print(f"  Skor XGB : {skor_xgb}/3")
print(f"  TERBAIK  : {nama_terbaik}")
print("  Catatan  : Random Forest Regressor digunakan sebagai model SPK.")


# ============================================================
# 7. SIMPAN MODEL RANDOM FOREST + LABEL ENCODER + METADATA
# ============================================================
print(f"\n[6/6] Menyimpan model Random Forest...")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Simpan Random Forest (model SPK)
with open(os.path.join(MODEL_DIR, "model_rf.pkl"), "wb") as f:
    pickle.dump(model_rf, f)

# Hapus model_xgb.pkl jika ada (dari versi sebelumnya)
xgb_path = os.path.join(MODEL_DIR, "model_xgb.pkl")
if os.path.isfile(xgb_path):
    os.remove(xgb_path)
    print(f"  [INFO] File lama {xgb_path} dihapus.")

# Simpan label encoder
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Simpan metadata — berisi evaluasi kedua model (untuk halaman Tentang)
metadata = {
    "feature_cols"   : FEATURE_COLS,
    "target_cols"    : TARGET_COLS,
    "kelas_tanah"    : list(le.classes_),
    "model_aktif"    : "Random Forest Regressor",
    "model_terbaik"  : nama_terbaik,   # informatif (hasil skor)
    "skor"           : {"rf": int(skor_rf), "xgb": int(skor_xgb)},
    # Detail evaluasi kedua model — untuk ditampilkan di halaman Tentang
    "evaluasi"       : {
        "rf"  : eval_rf,
        "xgb" : eval_xgb,
    },
}
with open(os.path.join(MODEL_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("\n" + "=" * 65)
print("MODEL RANDOM FOREST BERHASIL DISIMPAN!")
print("=" * 65)
print(f"  -> {MODEL_DIR}/model_rf.pkl       (Random Forest Regressor)")
print(f"  -> {MODEL_DIR}/label_encoder.pkl  ({len(le.classes_)} kelas tanah)")
print(f"  -> {MODEL_DIR}/metadata.pkl       (info evaluasi kedua model)")
print(f"\nKelas tanah yang dikenal: {list(le.classes_)}")
print(f"\nModel yang digunakan di SPK: Random Forest Regressor")
print("\nLangkah berikutnya: jalankan 'python app.py' untuk menjalankan website")
print("=" * 65)
