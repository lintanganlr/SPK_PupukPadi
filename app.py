"""
============================================================
  FLASK APP - SPK PENENTUAN DOSIS PUPUK PADI (v3)
============================================================
Web aplikasi yang memuat model Random Forest Regressor
hasil dari train_model.py.

Cara menjalankan:
  1. Pastikan sudah menjalankan: python train_model.py
     (akan menghasilkan folder model/ berisi model_rf.pkl,
      label_encoder.pkl, metadata.pkl)
  2. Jalankan: python app.py
  3. Buka browser: http://127.0.0.1:5000
============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ============================================================
# LOAD MODEL RANDOM FOREST, LABEL ENCODER, DAN METADATA
# ============================================================
MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
RF_PATH      = os.path.join(MODEL_DIR, 'model_rf.pkl')
LE_PATH      = os.path.join(MODEL_DIR, 'label_encoder.pkl')
META_PATH    = os.path.join(MODEL_DIR, 'metadata.pkl')

model          = None     # Random Forest Regressor
label_encoder  = None
metadata       = None
MODEL_LOADED   = False

try:
    with open(RF_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LE_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)

    MODEL_LOADED = True
    print("=" * 60)
    print(f"  MODEL RANDOM FOREST BERHASIL DIMUAT")
    print(f"  - Model aktif   : Random Forest Regressor")
    print(f"  - Kelas tanah   : {metadata['kelas_tanah']}")
    print(f"  - Fitur input   : {metadata['feature_cols']}")
    print("=" * 60)

except FileNotFoundError:
    print("=" * 60)
    print("  PERINGATAN: File model belum tersedia!")
    print("  Jalankan dulu: python train_model.py")
    print("  Server tetap jalan, tapi /predict akan error.")
    print("=" * 60)


# ============================================================
# INFORMASI APLIKASI PUPUK (sesuai proposal Section 2.2.3)
# ============================================================
INFO_APLIKASI = {
    'petroganik_premium': {
        'nama'  : 'Petroganik Premium',
        'cara'  : 'Ditebarkan merata di permukaan tanah kemudian dibenamkan ke dalam tanah.',
        'waktu' : 'Diberikan 1 (satu) kali sebelum tanam, pada saat pengolahan tanah terakhir.',
    },
    'urea': {
        'nama'  : 'Urea Petro (N 46%)',
        'cara'  : 'Ditebarkan merata di permukaan tanah pada kondisi lembab atau tergenang dangkal.',
        'waktu' : 'Diberikan secara bertahap dalam 3 fase: (1) awal vegetatif, (2) anakan maksimum, (3) awal pembentukan malai.',
    },
    'sp36': {
        'nama'  : 'SP-36 (P2O5 36%)',
        'cara'  : 'Ditebarkan merata di permukaan tanah kemudian dibenamkan ke dalam tanah.',
        'waktu' : 'Diberikan 1 (satu) kali pada awal pertanaman, saat pengolahan tanah terakhir atau menjelang tanam.',
    },
    'kplus': {
        'nama'  : 'K-Plus (K2O 30% + Boron)',
        'cara'  : 'Ditebarkan merata di permukaan tanah pada kondisi lembab atau tergenang dangkal.',
        'waktu' : 'Diberikan pada fase vegetatif hingga awal fase generatif, ketika kebutuhan tanaman terhadap unsur K relatif tinggi.',
    },
}


# ============================================================
# ROUTES HALAMAN
# ============================================================
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/kalkulator')
def kalkulator():
    kelas_tanah = metadata['kelas_tanah'] if MODEL_LOADED else ['liat', 'lempung', 'berpasir']
    return render_template(
        'kalkulator.html',
        kelas_tanah=kelas_tanah,
    )


@app.route('/edukasi')
def edukasi():
    return render_template('edukasi.html')


@app.route('/tentang')
def tentang():
    if MODEL_LOADED:
        return render_template(
            'tentang.html',
            model_aktif=metadata.get('model_aktif', 'Random Forest Regressor'),
            model_terbaik=metadata.get('model_terbaik', 'Random Forest Regressor'),
            eval_rf=metadata['evaluasi']['rf'],
            eval_xgb=metadata['evaluasi']['xgb'],
            skor=metadata['skor'],
        )
    return render_template(
        'tentang.html',
        model_aktif='Random Forest Regressor',
        model_terbaik='Model belum dilatih',
        eval_rf=None, eval_xgb=None, skor=None,
    )


# ============================================================
# ENDPOINT PREDIKSI
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi dosis pupuk.

    Input JSON:
        ph, c_organik, n_total, p_total, k_total,
        curah_hujan, suhu, jenis_tanah, target_panen

    Output JSON:
        petroganik_premium, urea, sp36, kplus  (kg/ha)
        + nama_model, info_aplikasi
    """
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model belum tersedia. Jalankan terlebih dahulu: python train_model.py'
        }), 503

    try:
        data = request.get_json(force=True)

        # --- Validasi field wajib ---
        required_fields = [
            'ph', 'c_organik', 'n_total', 'p_total', 'k_total',
            'curah_hujan', 'suhu', 'jenis_tanah', 'target_panen'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Field "{field}" tidak ditemukan'}), 400

        # --- Encode jenis tanah pakai LabelEncoder dari training ---
        jenis_tanah_norm = str(data['jenis_tanah']).strip().lower()
        if jenis_tanah_norm in label_encoder.classes_:
            jenis_tanah_enc = int(label_encoder.transform([jenis_tanah_norm])[0])
            tanah_warning   = None
        else:
            fallback        = label_encoder.classes_[0]
            jenis_tanah_enc = int(label_encoder.transform([fallback])[0])
            tanah_warning   = (
                f"Jenis tanah '{data['jenis_tanah']}' tidak ada pada data latih, "
                f"diganti otomatis menjadi '{fallback}'."
            )

        # --- Susun fitur dengan urutan SAMA dengan training (PENTING!) ---
        feature_cols = metadata['feature_cols']
        features_df = pd.DataFrame([{
            'pH'                  : float(data['ph']),
            'C_Organik'           : float(data['c_organik']),
            'N_Total'             : float(data['n_total']),
            'P_Total'             : float(data['p_total']),
            'K_Total'             : float(data['k_total']),
            'Target_Panen'        : float(data['target_panen']),
            'Curah_Hujan'         : float(data['curah_hujan']),
            'Suhu'                : float(data['suhu']),
            'Jenis_Tanah_Encoded' : jenis_tanah_enc,
        }])[feature_cols]   # paksa urutan kolom sesuai training

        # --- Prediksi pakai Random Forest Regressor ---
        predictions = model.predict(features_df)

        # Kliping nilai negatif (dosis tidak mungkin < 0)
        predictions = np.clip(predictions, 0, None)

        # --- Floor constraint: dosis brosur Petro = dosis MINIMAL ---
        # Berdasarkan konfirmasi Divisi Riset PT Petrokimia Gresik (Lampiran 1):
        # "dosis di brosur itu dosis minimal, jadi jangan ada yg di bawah itu".
        # Urutan kolom mengikuti target training: [P_Organik_Prem, Urea, SP36, Kplus]
        DOSIS_MINIMAL = [500, 300, 125, 75]   # Petroganik, Urea, SP-36, K-Plus
        for i, vmin in enumerate(DOSIS_MINIMAL):
            if vmin is not None and predictions[0][i] < vmin:
                predictions[0][i] = vmin

        # --- Susun response ---
        # Urutan target di training: P_Organik_Prem, Urea, SP36, Kplus
        result = {
            "petroganik_premium" : round(float(predictions[0][0]), 2),
            "urea"               : round(float(predictions[0][1]), 2),
            "sp36"               : round(float(predictions[0][2]), 2),
            "kplus"              : round(float(predictions[0][3]), 2),
            "nama_model"         : "Random Forest Regressor",
            "info_aplikasi"      : INFO_APLIKASI,
        }
        if tanah_warning:
            result['warning'] = tanah_warning

        return jsonify(result)

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Input tidak valid: {str(e)}'}), 422
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan server: {str(e)}'}), 500


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
