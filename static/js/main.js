/* =====================================================
   AgriSmart SPK — main.js  (v3)
   Handles:
     - Mobile navbar toggle
     - Navbar scroll behavior
     - Form validation (client-side)
     - Fetch POST /predict  (Random Forest Regressor)
     - Loading state & result rendering
     - Reset button
     - Deteksi cuaca otomatis (Open-Meteo)
   ===================================================== */

// ─────────────────────────────────────────
// NAVBAR — Mobile Toggle
// ─────────────────────────────────────────
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const mobileMenu    = document.getElementById('mobile-menu');

if (mobileMenuBtn && mobileMenu) {
  mobileMenuBtn.addEventListener('click', () => {
    const isHidden = mobileMenu.classList.toggle('hidden');
    mobileMenuBtn.setAttribute('aria-expanded', String(!isHidden));
  });

  // Close mobile menu when a link is clicked
  mobileMenu.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => mobileMenu.classList.add('hidden'));
  });
}

// ─────────────────────────────────────────
// NAVBAR — Scroll Shadow
// ─────────────────────────────────────────
const navbar = document.getElementById('navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    if (window.scrollY > 10) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }, { passive: true });
}


// ─────────────────────────────────────────
// KALKULATOR — Form Validation & Fetch
// ─────────────────────────────────────────
const form = document.getElementById('form-kalkulator');
if (form) {

  // ---- Validation Rules ----
  const validationRules = {
    ph:          { min: 0,   max: 14,   label: 'pH Tanah' },
    c_organik:   { min: 0,              label: 'C-Organik' },
    n_total:     { min: 0,              label: 'N-Total' },
    p_total:     { min: 0,              label: 'P-Total' },
    k_total:     { min: 0,              label: 'K-Total' },
    curah_hujan: { min: 0,              label: 'Curah Hujan' },
    suhu:        {                      label: 'Suhu' },
    target_panen:{ min: 0.01,           label: 'Target Panen' },
  };

  /**
   * Validates a single input field.
   * Returns an error message string or '' if valid.
   */
  function validateField(id, value) {
    const rule = validationRules[id];
    if (!rule) return '';

    if (value === '' || value === null || isNaN(Number(value))) {
      return `${rule.label} wajib diisi dengan angka.`;
    }
    const num = parseFloat(value);
    if (rule.min !== undefined && num < rule.min) {
      return `${rule.label} tidak boleh kurang dari ${rule.min}.`;
    }
    if (rule.max !== undefined && num > rule.max) {
      return `${rule.label} tidak boleh lebih dari ${rule.max}.`;
    }
    return '';
  }

  /**
   * Shows or clears an error on a field.
   */
  function setFieldError(id, message) {
    const input = document.getElementById(id);
    const errorEl = document.getElementById(`${id}-error`);
    if (!input) return;

    if (message) {
      input.classList.add('input-error');
      if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.remove('hidden');
      }
    } else {
      input.classList.remove('input-error');
      if (errorEl) {
        errorEl.textContent = '';
        errorEl.classList.add('hidden');
      }
    }
  }

  // Live validation on blur
  Object.keys(validationRules).forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('blur', () => {
        setFieldError(id, validateField(id, el.value));
      });
      el.addEventListener('input', () => {
        if (el.classList.contains('input-error')) {
          setFieldError(id, validateField(id, el.value));
        }
      });
    }
  });

  // Validate jenis_tanah (select)
  const jenisTanahEl = document.getElementById('jenis_tanah');
  if (jenisTanahEl) {
    jenisTanahEl.addEventListener('change', () => {
      setFieldError('jenis_tanah', jenisTanahEl.value ? '' : 'Jenis tanah wajib dipilih.');
    });
  }


  // ---- Helper: kirim payload ke /predict & render hasilnya ----
  async function callPredict(payload) {
    const loadingEl     = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const btnSubmit     = document.getElementById('btn-submit');

    loadingEl.classList.remove('hidden');
    resultSection.classList.add('hidden');
    if (btnSubmit)  btnSubmit.disabled  = true;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();

      // ── Render hasil ke result card ─────────────────────
      document.getElementById('res-petroganik').innerText = formatDosis(data.petroganik_premium);
      document.getElementById('res-urea').innerText       = formatDosis(data.urea);
      document.getElementById('res-sp36').innerText       = formatDosis(data.sp36);
      document.getElementById('res-kplus').innerText      = formatDosis(data.kplus);

      // ── Warning jenis tanah (kalau ada) ─────────────────
      if (data.warning) {
        showToast(`⚠️ ${data.warning}`, 'success');
      }

      // ── Tampilkan section hasil ─────────────────────────
      loadingEl.classList.add('hidden');
      resultSection.classList.remove('hidden');

      // ── Scroll ke hasil ──────────────────────────────────
      resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
      loadingEl.classList.add('hidden');
      showToast(`❌ Gagal mendapatkan prediksi: ${error.message}`, 'error');
      console.error('Predict error:', error);
    } finally {
      if (btnSubmit)  btnSubmit.disabled  = false;
    }
  }


  // ---- Form Submit ----
  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // --- Full validation before submit ---
    let hasError = false;

    Object.keys(validationRules).forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      const err = validateField(id, el.value);
      setFieldError(id, err);
      if (err) hasError = true;
    });

    // Validate select jenis_tanah
    const jtVal = jenisTanahEl ? jenisTanahEl.value : '';
    const jtErr = jtVal ? '' : 'Jenis tanah wajib dipilih.';
    setFieldError('jenis_tanah', jtErr);
    if (jtErr) hasError = true;

    if (hasError) {
      const firstError = form.querySelector('.input-error');
      if (firstError) {
        firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        firstError.focus();
      }
      return;
    }

    // --- Build payload (selalu pakai Random Forest) ---
    const payload = {
      ph:           parseFloat(document.getElementById('ph').value),
      c_organik:    parseFloat(document.getElementById('c_organik').value),
      n_total:      parseFloat(document.getElementById('n_total').value),
      p_total:      parseFloat(document.getElementById('p_total').value),
      k_total:      parseFloat(document.getElementById('k_total').value),
      curah_hujan:  parseFloat(document.getElementById('curah_hujan').value),
      suhu:         parseFloat(document.getElementById('suhu').value),
      jenis_tanah:  document.getElementById('jenis_tanah').value,
      target_panen: parseFloat(document.getElementById('target_panen').value),
    };

    await callPredict(payload);
  });


  // ─── Reset Button ───
  const btnReset = document.getElementById('btn-reset');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      form.reset();

      // Clear all errors
      Object.keys(validationRules).forEach(id => setFieldError(id, ''));
      setFieldError('jenis_tanah', '');

      // Hide result
      const resultSection = document.getElementById('result-section');
      if (resultSection) resultSection.classList.add('hidden');

      // Scroll back to form
      form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

} // end if (form)


// ─────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────

/**
 * Format angka dosis: tambahkan pemisah ribuan + satuan.
 * Contoh: 1250.5 → "1.250,5 kg/ha"
 */
function formatDosis(value) {
  if (value === undefined || value === null) return '— kg/ha';
  return Number(value).toLocaleString('id-ID', {
    maximumFractionDigits: 2,
  }) + ' kg/ha';
}

/**
 * Simple toast notification.
 * type: 'error' | 'success'
 */
function showToast(message, type = 'error') {
  const existing = document.getElementById('spk-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'spk-toast';
  toast.style.cssText = `
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    z-index: 9999; padding: 14px 24px; border-radius: 16px;
    font-size: 14px; font-weight: 600; box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    max-width: 90vw; text-align: center;
    background: ${type === 'error' ? '#fee2e2' : '#d1fae5'};
    color: ${type === 'error' ? '#991b1b' : '#065f46'};
    border: 1.5px solid ${type === 'error' ? '#fca5a5' : '#6ee7b7'};
    animation: slideUp 0.3s ease-out;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.4s';
    setTimeout(() => toast.remove(), 400);
  }, 5000);
}


// ─────────────────────────────────────────
// DETEKSI CUACA OTOMATIS
// (Open-Meteo API + Nominatim OSM — gratis, tanpa API key)
// ─────────────────────────────────────────
const btnDetectCuaca = document.getElementById('btn-detect-cuaca');

if (btnDetectCuaca) {

  function setStatus(message, type = 'info') {
    const el = document.getElementById('cuaca-status');
    if (!el) return;

    const colors = {
      info    : 'bg-sky-100 text-sky-800 border-sky-200',
      loading : 'bg-amber-50 text-amber-800 border-amber-200',
      success : 'bg-green-50 text-green-800 border-green-200',
      error   : 'bg-red-50 text-red-700 border-red-200',
    };

    el.className = `mt-3 text-sm px-4 py-2.5 rounded-xl border ${colors[type] || colors.info}`;
    el.innerHTML = message;
    el.classList.remove('hidden');
  }

  function setBtnLoading(loading) {
    const btnText = document.getElementById('btn-detect-text');
    btnDetectCuaca.disabled = loading;
    if (btnText) {
      btnText.textContent = loading
        ? 'Mengambil data cuaca…'
        : 'Deteksi Lokasi & Ambil Data Cuaca';
    }
  }

  function getKoordinat() {
    return new Promise((resolve, reject) => {
      if (!('geolocation' in navigator)) {
        reject(new Error('Browser Anda tidak mendukung Geolocation API.'));
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
        }),
        (err) => {
          const msgMap = {
            1: 'Izin lokasi ditolak. Anda bisa mengisi suhu & curah hujan secara manual.',
            2: 'Lokasi tidak dapat ditentukan. Coba beberapa saat lagi.',
            3: 'Permintaan lokasi timeout. Periksa koneksi internet Anda.',
          };
          reject(new Error(msgMap[err.code] || 'Gagal mendapatkan lokasi.'));
        },
        { enableHighAccuracy: false, timeout: 10000, maximumAge: 600000 }
      );
    });
  }

  async function getNamaLokasi(lat, lon) {
    try {
      const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10&accept-language=id`;
      const r = await fetch(url, {
        headers: { 'Accept': 'application/json' },
      });
      if (!r.ok) throw new Error('reverse geocode failed');
      const data = await r.json();
      const a = data.address || {};
      const lokasi = a.city || a.town || a.village || a.county || a.state || data.display_name || 'Lokasi tidak diketahui';
      const provinsi = a.state || '';
      return provinsi && lokasi !== provinsi ? `${lokasi}, ${provinsi}` : lokasi;
    } catch (e) {
      return `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
    }
  }

  async function getCuaca(lat, lon) {
    const url = 'https://api.open-meteo.com/v1/forecast'
      + `?latitude=${lat}&longitude=${lon}`
      + '&current=temperature_2m'
      + '&daily=precipitation_sum'
      + '&past_days=30'
      + '&forecast_days=1'
      + '&timezone=auto';

    const r = await fetch(url);
    if (!r.ok) throw new Error(`Open-Meteo error: ${r.status}`);
    const data = await r.json();

    const suhu = data.current && typeof data.current.temperature_2m === 'number'
      ? data.current.temperature_2m
      : null;

    let hujanTotal = 0;
    if (data.daily && Array.isArray(data.daily.precipitation_sum)) {
      const arr = data.daily.precipitation_sum.slice(0, 30).filter(v => typeof v === 'number');
      hujanTotal = arr.reduce((s, v) => s + v, 0);
    }

    if (suhu === null) throw new Error('Data suhu tidak tersedia dari API.');

    return {
      suhu      : Math.round(suhu * 10) / 10,
      curahHujan: Math.round(hujanTotal * 10) / 10,
    };
  }

  async function jalankanDeteksiCuaca() {
    setBtnLoading(true);
    setStatus('📍 Meminta izin lokasi… (klik <strong>Allow</strong> pada popup browser)', 'loading');

    try {
      const { lat, lon } = await getKoordinat();
      setStatus('🌍 Lokasi terdeteksi, mengambil data cuaca…', 'loading');

      const [namaLokasi, cuaca] = await Promise.all([
        getNamaLokasi(lat, lon),
        getCuaca(lat, lon),
      ]);

      const inputSuhu = document.getElementById('suhu');
      const inputHujan = document.getElementById('curah_hujan');
      if (inputSuhu)  inputSuhu.value  = cuaca.suhu;
      if (inputHujan) inputHujan.value = cuaca.curahHujan;

      [inputSuhu, inputHujan].forEach(el => {
        if (el) el.dispatchEvent(new Event('input', { bubbles: true }));
      });

      document.getElementById('chip-lokasi').textContent = namaLokasi;
      document.getElementById('chip-suhu').textContent   = cuaca.suhu.toFixed(1);
      document.getElementById('chip-hujan').textContent  = cuaca.curahHujan.toFixed(1);
      document.getElementById('cuaca-chips').classList.remove('hidden');

      setStatus(
        `✅ Berhasil! Suhu <strong>${cuaca.suhu.toFixed(1)} °C</strong> dan curah hujan
         <strong>${cuaca.curahHujan.toFixed(1)} mm</strong> (akumulasi 30 hari) telah diisi otomatis.
         Anda tetap bisa mengubahnya secara manual jika perlu.`,
        'success'
      );

    } catch (err) {
      console.error('Deteksi cuaca error:', err);
      setStatus(
        `⚠️ ${err.message || 'Terjadi kesalahan saat mengambil data cuaca.'}
         <br><span class="text-xs">Silakan isi suhu &amp; curah hujan secara manual.</span>`,
        'error'
      );
    } finally {
      setBtnLoading(false);
    }
  }

  btnDetectCuaca.addEventListener('click', jalankanDeteksiCuaca);
}
