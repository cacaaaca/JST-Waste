import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# --- 1. FUNGSI UNTUK MEMUAT MODEL (.PKL) ---
@st.cache_resource # Menggunakan cache agar model tidak di-load terus-menerus setiap kali ada interaksi
def load_model_app(model_path="model_jst_waste.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# --- 2. FUNGSI EKSTRAKSI FITUR DARI MATRIKS GAMBAR (OPENCV) ---
# Modifikasi dari fungsi extract_features_app agar menerima matriks gambar langsung, bukan path string
def extract_features_from_image(img, small_width=20, small_height=30):
    if img is None:
        raise ValueError("Matriks gambar kosong atau tidak valid.")

    h, w = img.shape[:2]

    # Gambar kecil (<= 20x30): flatten langsung
    if w <= small_width and h <= small_height:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (gray.astype(np.float32) / 255.0).flatten()

    # Gambar normal (> 20x30): ekstraksi fitur gabungan
    img  = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Hu Moments
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments   = cv2.moments(thresh)
    hu        = cv2.HuMoments(moments).flatten()
    hu        = np.array([-np.sign(v) * np.log10(abs(v) + 1e-10) for v in hu], dtype=np.float32)

    # Hue Channel
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hue_mean = np.array([np.mean(h_ch)], dtype=np.float32)
    hue_std  = np.array([np.std(h_ch)],  dtype=np.float32)
    hist_hue = cv2.calcHist([h_ch], [0], None, [8], [0, 180]).flatten()
    hist_hue = (hist_hue / (np.sum(hist_hue) + 1e-10)).astype(np.float32)
    sv_mean  = np.array([np.mean(s_ch), np.mean(v_ch)], dtype=np.float32)

    # LBP
    lbp      = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    lbp_hist = lbp_hist.astype(np.float32)

    # Tekstur Tambahan
    edges        = cv2.Canny(gray, 100, 200)
    edge_density = np.array([np.sum(edges > 0) / edges.size], dtype=np.float32)
    lap_var      = np.array([cv2.Laplacian(gray, cv2.CV_64F).var()], dtype=np.float32)
    gray_hist    = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    gray_hist    = (gray_hist / (np.sum(gray_hist) + 1e-10)).astype(np.float32)
    p            = gray_hist + 1e-10
    entropy      = np.array([-np.sum(p * np.log2(p))], dtype=np.float32)
    contrast     = np.array([np.std(gray)], dtype=np.float32)

    return np.concatenate([
        hu, hue_mean, hue_std, hist_hue, sv_mean,
        lbp_hist, gray_hist, edge_density, lap_var, entropy, contrast
    ]).astype(np.float32)

# --- 3. FUNGSI FEEDFORWARD JST ---
def feedforward_app(X, W1, b1, W2, b2, W3=None, b3=None):
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1) # ReLU
    z2 = np.dot(a1, W2) + b2
    
    # Kondisional check jika model menggunakan 2 Hidden Layer (Cara 2) atau 1 Hidden Layer
    if W3 is not None and b3 is not None:
        a2 = np.maximum(0, z2) # ReLU untuk layer 2
        z3 = np.dot(a2, W3) + b3
        out = z3
    else:
        out = z2
        
    return 1 / (1 + np.exp(-np.clip(out, -50, 50))) # Sigmoid

# --- 4. ANTARMUKA (UI) STREAMLIT ---
st.set_page_config(page_title="Waste Classification JST", page_icon="♻️")
st.title("♻️ Klasifikasi Sampah Menggunakan JST")
st.write("Ambil foto sampah organik atau anorganik (recyclable) langsung menggunakan kamera perangkat Anda.")

try:
    # Memuat model pkl yang tersimpan
    model = load_model_app("model_jst_waste.pkl")
    st.sidebar.success("✅ Model JST Berhasil Dimuat!")
    
    # Menampilkan info arsitektur model di sidebar sebagai pelengkap laporan
    st.sidebar.markdown("### 📋 Detail Arsitektur:")
    st.sidebar.write(f"- Fitur Input: {model.get('input_size', 49)}")
    if "W3" in model:
        st.sidebar.write(f"- Struktur Layer: {model['input_size']} ➔ {model['hidden_size1']} ➔ {model['hidden_size2']} ➔ 1")
    else:
        st.sidebar.write(f"- Struktur Layer: {model['input_size']} ➔ {model['hidden_size']} ➔ 1")
        
except Exception as e:
    st.error(f"Gagal memuat model (.pkl). Pastikan file model ada di folder yang sama. Error: {e}")
    st.stop()

# Menambahkan tab pilihan input: Kamera atau Upload File Gambar
tab1, tab2 = st.tabs(["📷 Kamera Langsung", "📁 Upload Gambar"])

with tab1:
    picture = st.camera_input("Arahkan kamera ke objek sampah")

with tab2:
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

# Satukan penanganan data dari Kamera maupun dari Upload File
target_image = None
if picture is not None:
    target_image = picture
elif uploaded_file is not None:
    target_image = uploaded_file

# Jika ada gambar yang masuk (dari salah satu opsi di atas)
if target_image is not None:
    # Konversi byte stream gambar dari Streamlit menjadi matriks BGR OpenCV
    bytes_data = target_image.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    img_opencv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_opencv is not None:
        st.info("🔄 Sedang mengekstrak fitur dan memproses klasifikasi...")
        
        try:
            # 1. Ekstrak fitur dari matriks gambar
            features = extract_features_from_image(img_opencv, model["small_width"], model["small_height"])
            
            # 2. Normalisasi fitur menggunakan StandardScaler bawaan model .pkl
            features_scaled = model["scaler"].transform([features])
            
            # 3. Feedforward JST (Otomatis handle jika model kamu memiliki W3/b3 atau tidak)
            W1, b1 = model["W1"], model["b1"]
            W2, b2 = model["W2"], model["b2"]
            W3 = model.get("W3", None)
            b3 = model.get("b3", None)
            
            prob = feedforward_app(features_scaled, W1, b1, W2, b2, W3, b3)[0][0]
            
            # 4. Penentuan kelas berdasarkan ambang batas (threshold) terbaik hasil tuning
            pred_idx = int(prob >= model["threshold"])
            label_hasil = model["output_names"][pred_idx]
            
            # 5. Tampilkan Hasil Prediksi ke Layar Web
            st.write("---")
            st.subheader("📊 Hasil Prediksi JST:")
            
            # Menampilkan indikator warna berdasarkan kelas hasil klasifikasi
            if label_hasil == "Organic":
                st.success(f"### Kategori: **{label_hasil}** (Sampah Organik)")
            else:
                st.warning(f"### Kategori: **{label_hasil}** (Sampah Anorganik/Recyclable)")
                
            st.metric(label="Nilai Probabilitas (Recyclable jika >= Threshold)", value=f"{prob:.4f}")
            st.write(f"Ambang Batas (*Best Threshold*) Model: `{model['threshold']:.3f}`")
            
        except Exception as err:
            st.error(f"Terjadi kegagalan komputasi JST: {err}")