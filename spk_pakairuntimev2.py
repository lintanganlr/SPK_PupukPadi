import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib


matplotlib.use('Agg')                      # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ── Visualisasi Chart ──────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
WARNA_RF  = '#2C7BB6'   # biru  → Random Forest
WARNA_XGB = '#D7191C'   # merah → XGBoost
OUTPUT_DIR = 'visualisasi_hasil_evaluasi_v5'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. MENAMPILKAN DATASET
# ============================================================
print("=" * 65)
print("  SISTEM PENDUKUNG KEPUTUSAN REKOMENDASI DOSIS PUPUK SPK_V5")
print("  Random Forest Regressor  vs  XGBoost Regressor")
print("  Preprocessing : Imputasi median (numerik) + modus (kategorik)")
print("=" * 65)

# Mencari file dataset di direktori yang sama dengan skrip
FILE_PATH = r"dataset_ujitanah.xlsx"
if not os.path.isfile(FILE_PATH):
    FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_ujitanah.xlsx"
    )

df = pd.read_excel(FILE_PATH)

print(f"\nDataset berhasil dimuat")
print(f"Jumlah baris awal : {df.shape[0]}")
print(f"Jumlah kolom      : {df.shape[1]}")
print(f"\nDataset 5 baris pertama:")
print(df.head())


# ============================================================
# 2. PREPROCESSING & PEMBERSIHAN DATA
# ============================================================
print("\n" + "-" * 65)
print("TAHAP PREPROCESSING")
print("-" * 65)

# --- 2a. Standarisasi nama kolom ---
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace('.', '_')
    .str.replace('-', '_')
)
print("Nama kolom setelah standarisasi:")
print(df.columns.tolist())

# --- 2b. Normalisasi kolom Jenis_Tanah ---
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
print(f"Nilai unik Jenis_Tanah : {df['Jenis_Tanah'].unique().tolist()}")

# --- 2c. Label Encoding pada Jenis_Tanah ---
le = LabelEncoder()
df['Jenis_Tanah_Encoded'] = le.fit_transform(df['Jenis_Tanah'])
print(
    f"Kelas Jenis_Tanah (encoded) : "
    f"{dict(zip(le.classes_, le.transform(le.classes_)))}"
)

# --- 2d. Konversi kolom numerik ke tipe data numerik ---
NUMERICAL_COLS = [
    'pH',
    'C_Organik',
    'N_Total',
    'P_Total',
    'K_Total',
    'Target_Panen',
    'Curah_Hujan',
    'Suhu',
    'P_Organik_Prem',
    'Urea',
    'SP36',
    'Kplus',
]
df[NUMERICAL_COLS] = df[NUMERICAL_COLS].apply(pd.to_numeric, errors='coerce')

# --- 2e. IMPUTASI MISSING VALUE (INTERPOLASI) ---
#         - Kolom numerik fitur input : diisi dengan MEDIAN (robust terhadap outlier)
#         - Kolom numerik target pupuk: diisi dengan MEDIAN (distribusi dosis bisa skewed)
#         - Kolom Jenis_Tanah_Encoded  : diisi dengan MODUS (nilai kategorik terbanyak)
print("\nPengecekan missing value sebelum imputasi:")
mv_sebelum = df[NUMERICAL_COLS + ['Jenis_Tanah_Encoded']].isnull().sum()
mv_ada = mv_sebelum[mv_sebelum > 0]
if mv_ada.empty:
    print("  Tidak ada missing value pada kolom yang digunakan.")
else:
    for col, jml in mv_ada.items():
        print(f"  {col:<22} : {jml} missing")

# Imputasi numerik dengan MEDIAN
FEATURE_NUM = ['pH', 'C_Organik', 'N_Total', 'P_Total', 'K_Total',
               'Target_Panen', 'Curah_Hujan', 'Suhu']
TARGET_NUM  = ['P_Organik_Prem', 'Urea', 'SP36', 'Kplus']

median_values = {}
for col in FEATURE_NUM + TARGET_NUM:
    med = df[col].median()
    median_values[col] = med
    jumlah_diisi = df[col].isnull().sum()
    if jumlah_diisi > 0:
        df[col] = df[col].fillna(med)
        print(f"  [MEDIAN] {col:<22} : {jumlah_diisi} nilai diisi dengan {med:.4f}")

# Imputasi Jenis_Tanah_Encoded dengan MODUS
modus_encoded = int(df['Jenis_Tanah_Encoded'].mode()[0])
jml_mv_encoded = df['Jenis_Tanah_Encoded'].isnull().sum()
if jml_mv_encoded > 0:
    df['Jenis_Tanah_Encoded'] = df['Jenis_Tanah_Encoded'].fillna(modus_encoded)
    nama_modus = le.inverse_transform([modus_encoded])[0]
    print(f"  [MODUS]  Jenis_Tanah_Encoded   : {jml_mv_encoded} nilai diisi "
          f"dengan {modus_encoded} ('{nama_modus}')")

print("\nPengecekan missing value setelah imputasi:")
mv_sesudah = df[NUMERICAL_COLS + ['Jenis_Tanah_Encoded']].isnull().sum().sum()
print(f"  Total missing value tersisa : {mv_sesudah}")
print(f"  Total baris setelah imputasi: {len(df)}")

# --- 2f. Hapus duplikat ---
dup = df.duplicated().sum()
df  = df.drop_duplicates().reset_index(drop=True)
print(f"Duplikat dihapus : {dup}")

# --- 2g. Statistik deskriptif data bersih ---
print(f"\nStatistik deskriptif data bersih:")
print(df[NUMERICAL_COLS].describe().round(4))


# visualisasi perhitungan skewness sama nilai imputasinya pada kolom yang missing value sebelum dan sesudah imputasi


# ============================================================
# 3. MENJELASKAN FITUR (X) DAN TARGET (y)
# ============================================================
print("\n" + "-" * 65)
print("MENJELASKAN FITUR DAN TARGET")
print("-" * 65)

FEATURE_COLS = [
    'pH', 'C_Organik', 'N_Total', 'P_Total', 'K_Total',
    'Target_Panen', 'Curah_Hujan', 'Suhu', 'Jenis_Tanah_Encoded',
]

TARGET_COLS = ['P_Organik_Prem', 'Urea', 'SP36', 'Kplus']

# Label ramah untuk visualisasi
TARGET_LABEL = {
    'P_Organik_Prem' : 'Petroganik\nPremium',
    'Urea'           : 'Urea',
    'SP36'           : 'SP-36',
    'Kplus'          : 'K-Plus',
}

FEATURE_LABEL = [
    'pH', 'C-Organik', 'N Total', 'P Total', 'K Total',
    'Target\nPanen', 'Curah\nHujan', 'Suhu', 'Jenis\nTanah',
]

X = df[FEATURE_COLS].copy()
y = df[TARGET_COLS].copy()

print(f"Fitur  (X) : {FEATURE_COLS}")
print(f"Target (y) : {TARGET_COLS}")
print(f"Shape  X   : {X.shape}")
print(f"Shape  y   : {y.shape}")


# ============================================================
# 4. SPLIT DATA LATIH DAN DATA UJI (80:20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nData latih : {X_train.shape[0]} sampel (80%)")
print(f"Data uji   : {X_test.shape[0]} sampel (20%)")


# ============================================================
# 5. FUNGSI EVALUASI MODEL
# ============================================================
def evaluasi_model(y_true, y_pred, target_cols):
    hasil = {}
    rmse_list, mae_list = [], []

    for i, col in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
        mae  = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
        r2   = r2_score(y_true.iloc[:, i], y_pred[:, i])
        hasil[col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        rmse_list.append(rmse)
        mae_list.append(mae)

    hasil['_rata_rata'] = {
        'RMSE' : np.mean(rmse_list),
        'MAE'  : np.mean(mae_list),
        'R2'   : r2_score(y_true, y_pred),
    }
    return hasil


def cetak_evaluasi(nama_model, evaluasi):
    print(f"\n{'=' * 65}")
    print(f"  HASIL EVALUASI : {nama_model}")
    print(f"{'=' * 65}")
    print(f"  {'Pupuk':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print(f"  {'-' * 52}")
    for col, m in evaluasi.items():
        if col == '_rata_rata':
            continue
        print(f"  {col:<20} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")
    avg = evaluasi['_rata_rata']
    print(f"  {'-' * 52}")
    print(f"  {'RATA-RATA':<20} {avg['RMSE']:>10.4f} {avg['MAE']:>10.4f} {avg['R2']:>10.4f}")
    print(f"  R2 Keseluruhan Model : {avg['R2']:.4f}")


# ============================================================
# 6. MODEL 1 — RANDOM FOREST REGRESSOR (dengan running time)
# ============================================================
print("\n" + "-" * 65)
print("  MELATIH MODEL 1 : Random Forest Regressor")
print("-" * 65)

rf_base = RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1,
)
model_rf = MultiOutputRegressor(rf_base)
waktu_mulai_rf  = time.time()
model_rf.fit(X_train, y_train)
waktu_rf        = time.time() - waktu_mulai_rf
print(f"Model Random Forest selesai dilatih.")
print(f"  Running time training RF   : {waktu_rf:.4f} detik")

y_pred_rf = model_rf.predict(X_test)
eval_rf   = evaluasi_model(y_test, y_pred_rf, TARGET_COLS)
cetak_evaluasi("Random Forest Regressor", eval_rf)


# ============================================================
# 7. MODEL 2 — XGBOOST REGRESSOR (dengan running time)
# ============================================================
print("\n" + "-" * 65)
print("MELATIH MODEL 2 : XGBoost Regressor")
print("-" * 65)

xgb_base = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0,
)
model_xgb = MultiOutputRegressor(xgb_base)
waktu_mulai_xgb = time.time()
model_xgb.fit(X_train, y_train)
waktu_xgb       = time.time() - waktu_mulai_xgb
print(f"Model XGBoost selesai dilatih.")
print(f"  Running time training XGB  : {waktu_xgb:.4f} detik")

y_pred_xgb = model_xgb.predict(X_test)
eval_xgb   = evaluasi_model(y_test, y_pred_xgb, TARGET_COLS)
cetak_evaluasi("XGBoost Regressor", eval_xgb)


# ============================================================
# 8. PERBANDINGAN & PEMILIHAN ALGORITMA TERBAIK
#    Sistem skor 4 metrik (RMSE, MAE, R², Running Time)
# ============================================================
print("\n" + "=" * 65)
print("PERBANDINGAN HASIL EVALUASI KEDUA MODEL")
print("=" * 65)

rmse_rf,  mae_rf,  r2_rf  = (eval_rf['_rata_rata']['RMSE'],
                               eval_rf['_rata_rata']['MAE'],
                               eval_rf['_rata_rata']['R2'])
rmse_xgb, mae_xgb, r2_xgb = (eval_xgb['_rata_rata']['RMSE'],
                               eval_xgb['_rata_rata']['MAE'],
                               eval_xgb['_rata_rata']['R2'])

tabel_perbandingan = pd.DataFrame({
    'Metrik'        : ['RMSE (rata-rata)', 'MAE (rata-rata)', 'R2 (keseluruhan)',
                       'Running Time Training (detik)*'],
    'Random Forest' : [rmse_rf,  mae_rf,  r2_rf,  round(waktu_rf, 4)],
    'XGBoost'       : [rmse_xgb, mae_xgb, r2_xgb, round(waktu_xgb, 4)],
    'Lebih Baik'    : [
        'XGBoost' if rmse_xgb < rmse_rf else 'Random Forest',
        'XGBoost' if mae_xgb  < mae_rf  else 'Random Forest',
        'XGBoost' if r2_xgb   > r2_rf   else 'Random Forest',
        'XGBoost' if waktu_xgb < waktu_rf else 'Random Forest',
    ],
})
tabel_perbandingan[['Random Forest', 'XGBoost']] = (
    tabel_perbandingan[['Random Forest', 'XGBoost']].round(4)
)
print(tabel_perbandingan.to_string(index=False))
print('* Running Time ditampilkan sebagai informasi tambahan, tidak masuk sistem skor.')

# Ringkasan running time
print(f"\n[RUNNING TIME]")
print(f"  Random Forest : {waktu_rf:.4f} detik")
print(f"  XGBoost       : {waktu_xgb:.4f} detik")
lebih_cepat = 'XGBoost' if waktu_xgb < waktu_rf else 'Random Forest'
selisih     = abs(waktu_rf - waktu_xgb)
print(f"  Model lebih cepat         : {lebih_cepat} (selisih {selisih:.4f} detik)")

# Sistem skor 3 metrik akurasi (RMSE rendah, MAE rendah, R2 tinggi)
# Running Time dicatat sebagai informasi tambahan, TIDAK masuk sistem skor
skor_rf  = (int(rmse_rf  < rmse_xgb) +
            int(mae_rf   < mae_xgb)  +
            int(r2_rf    > r2_xgb))
skor_xgb = 3 - skor_rf

if skor_rf > skor_xgb:
    nama_terbaik   = "Random Forest Regressor"
    model_terbaik  = model_rf
    eval_terbaik   = eval_rf
    y_pred_terbaik = y_pred_rf
elif skor_xgb > skor_rf:
    nama_terbaik   = "XGBoost Regressor"
    model_terbaik  = model_xgb
    eval_terbaik   = eval_xgb
    y_pred_terbaik = y_pred_xgb
else:
    # Tie-breaker: R2 tertinggi
    if r2_rf >= r2_xgb:
        nama_terbaik   = "Random Forest Regressor"
        model_terbaik  = model_rf
        eval_terbaik   = eval_rf
        y_pred_terbaik = y_pred_rf
    else:
        nama_terbaik   = "XGBoost Regressor"
        model_terbaik  = model_xgb
        eval_terbaik   = eval_xgb
        y_pred_terbaik = y_pred_xgb

print(f"\n[SKOR] Random Forest : {skor_rf}/3  |  XGBoost : {skor_xgb}/3  (berdasarkan 3 metrik akurasi: RMSE, MAE, R2)")  # berdasarkan 3 metrik akurasi: RMSE, MAE, R2
print(f"[KESIMPULAN] Algoritma terbaik : {nama_terbaik}")
print(f"             RMSE  RF:{rmse_rf:.4f}  XGB:{rmse_xgb:.4f}")
print(f"             MAE   RF:{mae_rf:.4f}   XGB:{mae_xgb:.4f}")
print(f"             R2    RF:{r2_rf:.4f}   XGB:{r2_xgb:.4f}")
print(f"\n{nama_terbaik} digunakan dalam Sistem Pendukung Keputusan.")



# ============================================================
# 9. DETAIL EVALUASI MODEL TERBAIK PER PUPUK
# ============================================================
print("\n" + "=" * 65)
print(f"  DETAIL EVALUASI MODEL TERBAIK : {nama_terbaik}")
print("=" * 65)
print(f"\n  {'Pupuk':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
print(f"  {'-' * 52}")
for col in TARGET_COLS:
    m = eval_terbaik[col]
    print(f"  {col:<20} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")
avg = eval_terbaik['_rata_rata']
print(f"  {'-' * 52}")
print(f"  {'RATA-RATA':<20} {avg['RMSE']:>10.4f} {avg['MAE']:>10.4f} {avg['R2']:>10.4f}")

waktu_terbaik = waktu_rf if nama_terbaik == "Random Forest Regressor" else waktu_xgb
print(f"\n  Running time training      : {waktu_terbaik:.4f} detik")


# ============================================================
# 10. VISUALISASI MODEL — 10 GRAFIK EVALUASI
# ============================================================
print("\n" + "=" * 65)
print("  MEMBUAT VISUALISASI MODEL")
print("=" * 65)

pupuk_labels  = [TARGET_LABEL[c] for c in TARGET_COLS]
x_pos         = np.arange(len(TARGET_COLS))
BAR_W         = 0.35
warna_terbaik = WARNA_RF if nama_terbaik == "Random Forest Regressor" else WARNA_XGB


# ── Viz 1: Perbandingan RMSE per Pupuk ──────────────────────
rmse_rf_per  = [eval_rf[c]['RMSE']  for c in TARGET_COLS]
rmse_xgb_per = [eval_xgb[c]['RMSE'] for c in TARGET_COLS]

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x_pos - BAR_W/2, rmse_rf_per,  BAR_W, color=WARNA_RF,
            label='Random Forest', edgecolor='white', linewidth=0.6)
b2 = ax.bar(x_pos + BAR_W/2, rmse_xgb_per, BAR_W, color=WARNA_XGB,
            label='XGBoost', edgecolor='white', linewidth=0.6)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(pupuk_labels, fontsize=10)
ax.set_ylabel('RMSE (kg/ha)', fontsize=11)
ax.set_title('Perbandingan RMSE per Jenis Pupuk\nRandom Forest vs XGBoost Regressor',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(rmse_rf_per), max(rmse_xgb_per)) * 1.25)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '01_perbandingan_RMSE_per_pupuk.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] {path}")


# ── Viz 2: Perbandingan MAE per Pupuk ───────────────────────
mae_rf_per  = [eval_rf[c]['MAE']  for c in TARGET_COLS]
mae_xgb_per = [eval_xgb[c]['MAE'] for c in TARGET_COLS]

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x_pos - BAR_W/2, mae_rf_per,  BAR_W, color=WARNA_RF,
            label='Random Forest', edgecolor='white', linewidth=0.6)
b2 = ax.bar(x_pos + BAR_W/2, mae_xgb_per, BAR_W, color=WARNA_XGB,
            label='XGBoost', edgecolor='white', linewidth=0.6)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(pupuk_labels, fontsize=10)
ax.set_ylabel('MAE (kg/ha)', fontsize=11)
ax.set_title('Perbandingan MAE per Jenis Pupuk\nRandom Forest vs XGBoost Regressor',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(mae_rf_per), max(mae_xgb_per)) * 1.25)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '02_perbandingan_MAE_per_pupuk.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] {path}")


# ── Viz 3: Perbandingan R2 per Pupuk ────────────────────────
r2_rf_per  = [eval_rf[c]['R2']  for c in TARGET_COLS]
r2_xgb_per = [eval_xgb[c]['R2'] for c in TARGET_COLS]

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x_pos - BAR_W/2, r2_rf_per,  BAR_W, color=WARNA_RF,
            label='Random Forest', edgecolor='white', linewidth=0.6)
b2 = ax.bar(x_pos + BAR_W/2, r2_xgb_per, BAR_W, color=WARNA_XGB,
            label='XGBoost', edgecolor='white', linewidth=0.6)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.4f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold')

ax.axhline(1.0, linestyle='--', color='gray', linewidth=0.8, alpha=0.6,
           label='R2 = 1 (sempurna)')
ax.set_xticks(x_pos)
ax.set_xticklabels(pupuk_labels, fontsize=10)
ax.set_ylabel('R2 Score', fontsize=11)
ax.set_ylim(min(min(r2_rf_per), min(r2_xgb_per)) - 0.05, 1.06)
ax.set_title('Perbandingan R2 Score per Jenis Pupuk\nRandom Forest vs XGBoost Regressor',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '03_perbandingan_R2_per_pupuk.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] {path}")


# ── Viz 4: Dashboard Ringkasan 3 Metrik ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Ringkasan Perbandingan Evaluasi Model\nRandom Forest Regressor vs XGBoost Regressor',
    fontsize=13, fontweight='bold', y=1.02
)

metrik_data = [
    ('RMSE\n(rata-rata, kg/ha)', rmse_rf,  rmse_xgb, False),
    ('MAE\n(rata-rata, kg/ha)',  mae_rf,   mae_xgb,  False),
    ('R2\n(keseluruhan)',        r2_rf,    r2_xgb,   True),
]

for ax, (judul, val_rf, val_xgb, higher_better) in zip(axes, metrik_data):
    bars = ax.bar(['Random\nForest', 'XGBoost'], [val_rf, val_xgb],
                  color=[WARNA_RF, WARNA_XGB], width=0.45,
                  edgecolor='white', linewidth=0.8)

    pemenang_idx = (0 if (val_rf > val_xgb) == higher_better else 1)
    val_max      = max(val_rf, val_xgb)
    ax.text(pemenang_idx, val_max * 1.06, '* Terbaik',
            ha='center', fontsize=9, color='#e67e22', fontweight='bold')

    for bar, val in zip(bars, [val_rf, val_xgb]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() / 2,
                f'{val:.4f}',
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    ax.set_title(judul, fontsize=11, fontweight='bold')
    ax.set_ylim(0, val_max * 1.22)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=10)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '04_dashboard_ringkasan_metrik.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] {path}")


# ── Viz 5: Radar Chart Perbandingan Model ───────────────────
labels_radar = ['R2\nKeseluruhan', 'R2\nP.Organik', 'R2\nUrea',
                'R2\nSP-36', 'R2\nK-Plus',
                '1-RMSE\n(norm)', '1-MAE\n(norm)']
N = len(labels_radar)

raw_rf  = [r2_rf,
           eval_rf['P_Organik_Prem']['R2'],
           eval_rf['Urea']['R2'],
           eval_rf['SP36']['R2'],
           eval_rf['Kplus']['R2'],
           rmse_rf, mae_rf]
raw_xgb = [r2_xgb,
           eval_xgb['P_Organik_Prem']['R2'],
           eval_xgb['Urea']['R2'],
           eval_xgb['SP36']['R2'],
           eval_xgb['Kplus']['R2'],
           rmse_xgb, mae_xgb]

def norm_radar(vals_rf, vals_xgb):
    out_rf, out_xgb = [], []
    for i, (a, b) in enumerate(zip(vals_rf, vals_xgb)):
        if i < 5:
            out_rf.append(a); out_xgb.append(b)
        else:
            lo, hi = min(a, b), max(a, b)
            span = hi - lo if hi != lo else 1
            out_rf.append(1 - (a - lo) / span)
            out_xgb.append(1 - (b - lo) / span)
    return out_rf, out_xgb

val_rf_r, val_xgb_r = norm_radar(raw_rf, raw_xgb)
val_rf_r  += val_rf_r[:1]
val_xgb_r += val_xgb_r[:1]

angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, val_rf_r,  'o-', linewidth=2, color=WARNA_RF,  label='Random Forest')
ax.fill(angles, val_rf_r,  alpha=0.18, color=WARNA_RF)
ax.plot(angles, val_xgb_r, 's-', linewidth=2, color=WARNA_XGB, label='XGBoost')
ax.fill(angles, val_xgb_r, alpha=0.18, color=WARNA_XGB)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7, color='gray')
ax.set_title('Radar Chart Perbandingan Model\n(ternormalisasi: lebih luar = lebih baik)',
             fontsize=11, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '05_radar_chart_perbandingan_model.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] {path}")


# ============================================================
# 11. SISTEM PENDUKUNG KEPUTUSAN — FUNGSI REKOMENDASI PUPUK
# ============================================================
print("\n" + "=" * 65)
print("  SISTEM PENDUKUNG KEPUTUSAN REKOMENDASI DOSIS PUPUK")
print("  Menggunakan : Random Forest Regressor")
print("=" * 65)

# Informasi cara & waktu pengaplikasian — Sumber: Proposal Section 2.2.3
INFO_APLIKASI = {
    'P_Organik_Prem': {
        'nama'   : 'Petroganik Premium',
        'fungsi' : 'Pupuk organik – sumber C-Organik & pembenah tanah',
        'cara'   : ('Ditebarkan merata di permukaan tanah kemudian '
                    'dibenamkan/dicampurkan ke dalam tanah.'),
        'waktu'  : ('Diberikan 1 (satu) kali sebelum tanam, '
                    'pada saat pengolahan tanah terakhir.'),
    },
    'Urea': {
        'nama'   : 'Urea Petro (N 46%)',
        'fungsi' : 'Pupuk nitrogen – mendukung pertumbuhan vegetatif',
        'cara'   : ('Ditebarkan merata di permukaan tanah '
                    'pada kondisi lembab atau tergenang dangkal.'),
        'waktu'  : ('Diberikan secara bertahap (split application) '
                    'dalam 3 fase: (1) awal pertumbuhan vegetatif, '
                    '(2) pembentukan anakan maksimum, '
                    '(3) awal pembentukan malai.'),
    },
    'SP36': {
        'nama'   : 'SP-36 (P2O5 36%)',
        'fungsi' : 'Pupuk fosfor – mendukung pembentukan akar & fase awal',
        'cara'   : ('Ditebarkan merata di permukaan tanah '
                    'kemudian dibenamkan ke dalam tanah.'),
        'waktu'  : ('Diberikan 1 (satu) kali pada awal pertanaman, '
                    'yaitu saat pengolahan tanah terakhir '
                    'atau menjelang tanam.'),
    },
    'Kplus': {
        'nama'   : 'K-Plus (K2O 30% + Boron)',
        'fungsi' : 'Pupuk kalium – mendukung pembentukan anakan & malai',
        'cara'   : ('Ditebarkan merata di permukaan tanah '
                    'pada kondisi lembab atau tergenang dangkal.'),
        'waktu'  : ('Diberikan pada fase vegetatif hingga awal fase '
                    'generatif, ketika kebutuhan tanaman terhadap '
                    'unsur K relatif tinggi.'),
    },
}


def rekomendasi_pupuk(
    pH, c_organik, n_total, p_total, k_total,
    target_panen, curah_hujan, suhu, jenis_tanah,
    model, label_encoder, nama_model,
):
    """Memberikan rekomendasi dosis pupuk beserta cara & waktu pengaplikasian."""
    # Normalisasi & validasi jenis tanah
    jenis_tanah_norm = str(jenis_tanah).strip().lower()
    if jenis_tanah_norm not in label_encoder.classes_:
        fallback = label_encoder.classes_[0]
        print(f"  [WARNING] Jenis tanah '{jenis_tanah}' tidak dikenal, "
              f"diganti menjadi '{fallback}'.")
        jenis_tanah_norm = fallback
    jenis_tanah_enc = int(label_encoder.transform([jenis_tanah_norm])[0])

    # DataFrame input sesuai urutan FEATURE_COLS
    data_input = pd.DataFrame({
        'pH'                  : [pH],
        'C_Organik'           : [c_organik],
        'N_Total'             : [n_total],
        'P_Total'             : [p_total],
        'K_Total'             : [k_total],
        'Target_Panen'        : [target_panen],
        'Curah_Hujan'         : [curah_hujan],
        'Suhu'                : [suhu],
        'Jenis_Tanah_Encoded' : [jenis_tanah_enc],
    })

    # Prediksi & kliping nilai negatif
    hasil_pred = model.predict(data_input)
    hasil_df   = pd.DataFrame(hasil_pred, columns=TARGET_COLS)
    hasil_df   = hasil_df.clip(lower=0)

    # Floor constraint: dosis brosur Petro = dosis minimal (lihat Lampiran 1)
    DOSIS_MINIMAL = {'P_Organik_Prem': 500, 'Urea': 300, 'SP36': 125, 'Kplus': 75}
    for col, vmin in DOSIS_MINIMAL.items():
        if vmin is not None:
            hasil_df[col] = hasil_df[col].clip(lower=vmin)

    # ── Cetak hasil ──────────────────────────────────────────
    print(f"\n  {'=' * 58}")
    print(f"  INPUT PARAMETER LAHAN")
    print(f"  {'-' * 58}")
    print(f"  pH Tanah              : {pH}")
    print(f"  C-Organik (%)         : {c_organik}")
    print(f"  N Total (%)           : {n_total}")
    print(f"  P Total (ppm)         : {p_total}")
    print(f"  K Total (me/100g)     : {k_total}")
    print(f"  Target Panen (ton/ha) : {target_panen}")
    print(f"  Curah Hujan (mm/bln)  : {curah_hujan}")
    print(f"  Suhu (Celsius)        : {suhu}")
    print(f"  Jenis Tanah           : {jenis_tanah} (encoded: {jenis_tanah_enc})")

    print(f"\nREKOMENDASI DOSIS PUPUK — {nama_model}")
    print(f"  {'-' * 58}")
    print(f"  {'Pupuk':<22} {'Nama Produk':<20} {'Dosis':>12}")
    print(f"  {'-' * 58}")
    for col in TARGET_COLS:
        dosis       = hasil_df[col].values[0]
        nama_produk = INFO_APLIKASI[col]['nama']
        print(f"  {col:<22} {nama_produk:<20} {dosis:>9.2f} kg/ha")
    print(f"  {'-' * 58}")

    # ── Cara & Waktu Pengaplikasian ───────────────────────────
    print(f"\n  CARA & WAKTU PENGAPLIKASIAN PUPUK")
    for col in TARGET_COLS:
        info  = INFO_APLIKASI[col]
        dosis = hasil_df[col].values[0]
        print(f"\n  > {info['nama']}  ({dosis:.2f} kg/ha)")
        print(f"    Fungsi : {info['fungsi']}")
        print(f"    Cara   : {info['cara']}")
        print(f"    Waktu  : {info['waktu']}")
    print(f"  {'=' * 58}")

    return hasil_df


# ============================================================
# 12. INPUT MANUAL — SISTEM PENDUKUNG KEPUTUSAN INTERAKTIF
#     Range input disesuaikan dengan rentang nilai dataset_ujitanah.xlsx
# ============================================================

def input_float(prompt, min_val=None, max_val=None):
    """Meminta input angka desimal dari pengguna dengan validasi range."""
    while True:
        try:
            sys.stdout.flush()
            raw = input(prompt).strip()
            if raw == '':
                print("  [!] Input tidak boleh kosong. Masukkan angka (contoh: 5.5 atau 28).")
                continue
            nilai = float(raw)
            if min_val is not None and nilai < min_val:
                print(f"  [!] Nilai minimal adalah {min_val}. Silakan coba lagi.")
                continue
            if max_val is not None and nilai > max_val:
                print(f"  [!] Nilai maksimal adalah {max_val}. Silakan coba lagi.")
                continue
            return nilai
        except ValueError:
            print("  [!] Input tidak valid. Masukkan angka (contoh: 5.5 atau 28).")


def input_jenis_tanah(prompt, kelas_tersedia):
    """Meminta input jenis tanah dari pengguna."""
    print(f"\n  Jenis tanah tersedia di data latih : {', '.join(kelas_tersedia)}")
    print("  (Boleh input jenis tanah lain — sistem akan pakai fallback otomatis)")
    sys.stdout.flush()
    while True:
        nilai = input(prompt).strip().lower()
        if nilai == '':
            print("  [!] Jenis tanah tidak boleh kosong.")
            continue
        return nilai


def jalankan_input_manual():
    """
    Mengumpulkan parameter lahan dari pengguna.
    Range validasi disesuaikan dengan rentang nilai dataset_ujitanah.xlsx:
      pH         : 2.50 – 8.00     (median 5.85)
      C_Organik  : 0.10 – 4.70     (median 1.79)
      N_Total    : 0.02 – 0.43     (median 0.14)
      P_Total    : 18.04 – 680.85  (median 173.58)
      K_Total    : 11.03 – 573.72  (median 47.75)
      Target_Panen: 2.00 – 8.00    (median 5.00)
      Curah_Hujan: 50.00 – 350.00  (median 158.00)
      Suhu       : 21.00 – 32.00   (median 27.30)
    """
    print("\n" + "=" * 65)
    print("  INPUT DATA KONDISI LAHAN")
    print("  Isi setiap parameter sesuai hasil uji tanah Anda.")
    print("=" * 65)

    # --- Algoritma yang digunakan: Random Forest Regressor ---
    nama_model   = 'Random Forest Regressor'
    model_dipilih = model_rf
    print(f"\n  Algoritma digunakan : {nama_model}")

    pH          = input_float("  pH Tanah               (misal: 5.85 | range 2.5–8)    : ", 2.5, 8.0)
    c_organik   = input_float("  C-Organik (%)          (misal: 1.79 | range 0.1–5)    : ", 0.1, 5.0)
    n_total     = input_float("  N Total (%)            (misal: 0.14 | range 0.02–0.5) : ", 0.02, 0.5)
    p_total     = input_float("  P Total (ppm)          (misal: 173  | range 18–700)   : ", 18.0, 700.0)
    k_total     = input_float("  K Total (me/100g)      (misal: 47.75| range 11–600)   : ", 11.0, 600.0)
    target_panen= input_float("  Target Panen (ton/ha)  (misal: 5.0  | range 2–8)      : ", 2.0, 8.0)
    curah_hujan = input_float("  Curah Hujan (mm/bln)   (misal: 158  | range 50–350)   : ", 50.0, 350.0)
    suhu        = input_float("  Suhu (°C)              (misal: 27.3 | range 21–32)    : ", 21.0, 32.0)
    jenis_tanah = input_jenis_tanah(
        "  Jenis Tanah                                          : ",
        le.classes_,
    )

    hasil = rekomendasi_pupuk(
        pH=pH, c_organik=c_organik, n_total=n_total, p_total=p_total,
        k_total=k_total, target_panen=target_panen,
        curah_hujan=curah_hujan, suhu=suhu, jenis_tanah=jenis_tanah,
        model=model_dipilih, label_encoder=le, nama_model=nama_model,
    )

    # Simpan parameter input bersama hasil untuk keperluan ekspor
    hasil['pH']          = pH
    hasil['C_Organik']   = c_organik
    hasil['N_Total']     = n_total
    hasil['P_Total']     = p_total
    hasil['K_Total']     = k_total
    hasil['Target_Panen']= target_panen
    hasil['Curah_Hujan'] = curah_hujan
    hasil['Suhu']        = suhu
    hasil['Jenis_Tanah'] = jenis_tanah

    return hasil


# -- Loop utama: bisa prediksi berulang kali ------------------
semua_hasil = []
sesi        = 1

print("\n" + "=" * 65)
print("  SISTEM PENDUKUNG KEPUTUSAN MODE INPUT MANUAL")
print("  Model aktif : Random Forest Regressor")
print("=" * 65)

while True:
    print(f"\n{'-' * 65}")
    print(f"  SESI PREDIKSI KE-{sesi}")
    print(f"{'-' * 65}")

    hasil_sesi = jalankan_input_manual()
    semua_hasil.append(hasil_sesi)
    sesi += 1

    print("\n" + "-" * 65)
    lanjut = input("  Ingin melakukan prediksi lagi? (y/n) : ").strip().lower()
    if lanjut != 'y':
        print("\n  Keluar dari mode input. Melanjutkan ke ekspor Excel...")
        break


# ============================================================
# 13. EKSPOR HASIL EVALUASI KE EXCEL
# ============================================================
OUTPUT_EXCEL = "hasil_evaluasi_model_v5.xlsx"

# Susun ringkasan semua sesi prediksi manual
kolom_input  = ['pH','C_Organik','N_Total','P_Total','K_Total',
                'Target_Panen','Curah_Hujan','Suhu','Jenis_Tanah']
kolom_output = TARGET_COLS

rows_prediksi = []
for i, h in enumerate(semua_hasil, 1):
    baris = {'Sesi': i}
    for k in kolom_input:
        baris[k] = h[k].values[0] if k in TARGET_COLS else h[k]
    for k in kolom_output:
        baris[k + '_kg_ha'] = round(h[k].values[0], 4)
    rows_prediksi.append(baris)

df_prediksi = pd.DataFrame(rows_prediksi)

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

    # Sheet 1 — Perbandingan model (termasuk Running Time)
    tabel_perbandingan.to_excel(
        writer, sheet_name='Perbandingan_Model', index=False
    )

    # Sheet 2 — Detail evaluasi Random Forest per pupuk
    rows_rf = [{'Pupuk': c, 'RMSE': eval_rf[c]['RMSE'],
                'MAE': eval_rf[c]['MAE'], 'R2': eval_rf[c]['R2']}
               for c in TARGET_COLS]
    rows_rf.append({'Pupuk': 'RATA-RATA', **eval_rf['_rata_rata']})
    pd.DataFrame(rows_rf).round(4).to_excel(
        writer, sheet_name='Evaluasi_RandomForest', index=False
    )

    # Sheet 3 — Detail evaluasi XGBoost per pupuk
    rows_xgb = [{'Pupuk': c, 'RMSE': eval_xgb[c]['RMSE'],
                 'MAE': eval_xgb[c]['MAE'], 'R2': eval_xgb[c]['R2']}
                for c in TARGET_COLS]
    rows_xgb.append({'Pupuk': 'RATA-RATA', **eval_xgb['_rata_rata']})
    pd.DataFrame(rows_xgb).round(4).to_excel(
        writer, sheet_name='Evaluasi_XGBoost', index=False
    )

    # Sheet 4 — Riwayat semua prediksi manual sesi ini
    if not df_prediksi.empty:
        df_prediksi.to_excel(
            writer, sheet_name='Riwayat_Prediksi', index=False
        )

    # Sheet 5 — Ringkasan running time kedua model
    df_waktu = pd.DataFrame({
        'Model'                 : ['Random Forest Regressor', 'XGBoost Regressor'],
        'Running Time (detik)'  : [round(waktu_rf, 4), round(waktu_xgb, 4)],
        'Running Time (menit)'  : [round(waktu_rf / 60, 4), round(waktu_xgb / 60, 4)],
        'Lebih Cepat'           : [
            'Ya' if waktu_rf  < waktu_xgb else 'Tidak',
            'Ya' if waktu_xgb < waktu_rf  else 'Tidak',
        ],
        'Selisih (detik)'       : [
            round(abs(waktu_rf - waktu_xgb), 4),
            round(abs(waktu_rf - waktu_xgb), 4),
        ],
    })
    df_waktu.to_excel(writer, sheet_name='Running_Time', index=False)

print(f"\nHasil evaluasi diekspor ke : {OUTPUT_EXCEL}")
print("\n" + "=" * 65)
print("PROSES SELESAI")
print(f"File visualisasi : ./{OUTPUT_DIR}/  (10 grafik)")
print(f"File evaluasi    : {OUTPUT_EXCEL} (5 sheet)")
print(f"Total sesi prediksi manual : {len(semua_hasil)} sesi")
print("=" * 65)
