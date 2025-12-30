
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import io

# ## Dokumentasi: Import Library
# Mengimpor library yang diperlukan:
# - streamlit: untuk membuat antarmuka web
# - pandas/numpy: untuk manipulasi data
# - matplotlib/seaborn/plotly: untuk visualisasi data
# - sklearn/xgboost: untuk algoritma machine learning

# Config
# ## Dokumentasi: Konfigurasi Halaman
# Mengatur judul tab browser, icon, dan layout halaman menjadi 'wide' (lebar).
st.set_page_config(
    page_title="Academic Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Tidy" look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Dictionary Mappings ---
# ## Dokumentasi: Kamus Data (Data Dictionary)
# Memetakan kode angka pada kolom kategorikal menjadi teks deskriptif dalam Bahasa Indonesia.
# Digunakan untuk menampilkan label yang mudah dibaca user pada grafik dan tabel.
# Based on DOI:10.3390/data7110146
DATA_DICTIONARY = {
    "Marital status (Status Pernikahan)": {
        1: "Lajang", 2: "Menikah", 3: "Duda/Janda", 4: "Cerai", 5: "Kumpul Kebo (Facto Union)", 6: "Pisah ranjang sah"
    },
    "Application mode (Jalur Pendaftaran)": {
        1: "Tahap 1 - kontingen umum", 2: "Ordonansi No. 612/93", 5: "Tahap 1 - kontingen khusus (Azores)",
        7: "Pemegang gelar pendidikan tinggi lain", 10: "Ordonansi No. 854-B/99", 15: "Mahasiswa Internasional",
        16: "Tahap 1 - kontingen khusus (Madeira)", 17: "Tahap 2 - kontingen umum", 18: "Tahap 3 - kontingen umum",
        26: "Rencana berbeda (Ordonansi 533-A/99)", 27: "Institusi lain (Ordonansi 533-A/99)",
        39: "Di atas 23 tahun", 42: "Transfer", 43: "Pindah jurusan", 44: "Pemegang diploma spesialisasi teknologi",
        51: "Pindah institusi/jurusan", 53: "Pemegang diploma siklus pendek", 57: "Pindah institusi/jurusan (Internasional)"
    },
    "Course (Jurusan)": {
        33: "Teknologi Produksi Biofuel", 171: "Desain Animasi dan Multimedia", 
        8014: "Layanan Sosial (malam)", 9003: "Agronomi", 9070: "Desain Komunikasi", 
        9085: "Keperawatan Hewan", 9119: "Teknik Informatika", 9130: "Equinculture (Kuda)", 
        9147: "Manajemen", 9238: "Layanan Sosial", 9254: "Pariwisata", 9500: "Keperawatan", 
        9556: "Kebersihan Gigi (Oral Hygiene)", 9670: "Manajemen Periklanan dan Pemasaran", 
        9773: "Jurnalisme dan Komunikasi", 9853: "Pendidikan Dasar", 9991: "Manajemen (malam)"
    },
    "Daytime/evening attendance (Waktu Kuliah)": {
        1: "Siang (Daytime)", 0: "Malam (Evening)"
    },
    "Previous qualification (Kualifikasi Sebelumnya)": {
        1: "Pendidikan Menengah", 2: "Pendidikan Tinggi - sarjana muda", 3: "Pendidikan Tinggi - sarjana",
        4: "Pendidikan Tinggi - magister", 5: "Pendidikan Tinggi - doktor", 6: "Frekuensi pendidikan tinggi",
        9: "Kelas 12 - tidak tamat", 10: "Kelas 11 - tidak tamat",
        12: "Lainnya - Kelas 11", 14: "Kelas 10", 15: "Kelas 10 - tidak tamat",
        19: "Pendidikan dasar siklus ke-3", 38: "Pendidikan dasar siklus ke-2",
        39: "Kursus spesialisasi teknologi", 40: "Pendidikan Tinggi - sarjana (siklus 1)",
        42: "Kursus teknis pendidikan tinggi profesional", 43: "Pendidikan Tinggi - magister (siklus 2)"
    },
    "Nacionality (Kewarganegaraan)": {
        1: "Portugis", 2: "Jerman", 6: "Spanyol", 11: "Italia", 13: "Belanda", 14: "Inggris",
        17: "Lithuania", 21: "Angola", 22: "Tanjung Verde", 24: "Guinea", 25: "Mozambik",
        26: "Sao Tome", 32: "Turki", 41: "Brasil", 62: "Rumania", 100: "Moldova",
        101: "Meksiko", 103: "Ukraina", 105: "Rusia", 108: "Kuba", 109: "Kolombia"
    },
    "Gender (Jenis Kelamin)": {
        1: "Laki-laki", 0: "Perempuan"
    },
    "Scholarship holder (Penerima Beasiswa)": {
        1: "Ya", 0: "Tidak"
    },
    "International (Mahasiswa Internasional)": {
        1: "Ya", 0: "Tidak"
    },
    "Tuition fees up to date (SPP Lancar)": {
        1: "Ya", 0: "Tidak"
    },
    "Mother's qualification (Pendidikan Ibu)": {
        1: "Pendidikan Menengah (Kelas 12)", 2: "Sarjana Muda (Bachelor)", 3: "Sarjana (Degree)",
        4: "Magister (Master)", 5: "Doktor (PhD)", 6: "Frekuensi Pendidikan Tinggi",
        9: "Kelas 12 (Tidak Tamat)", 10: "Kelas 11 (Tidak Tamat)", 11: "Kelas 7 (Lama)",
        12: "Lainnya - Kelas 11", 14: "Kelas 10", 18: "Kursus Perdagangan Umum",
        19: "Pendidikan Dasar Siklus 3", 26: "Kelas 7", 27: "Kelas 6 (Siklus 2)",
        29: "Kelas 9 (Siklus 3)", 30: "Kursus Lanjutan (Siklus 3)", 34: "Tidak Ada",
        35: "Tidak Bisa Baca Tulis", 36: "Bisa Baca Tanpa Sekolah", 37: "Pendidikan Dasar Siklus 1",
        38: "Pendidikan Dasar Siklus 2", 39: "Kursus Spesialisasi Teknologi", 
        40: "Pendidikan Tinggi - Sarjana (Siklus 1)", 41: "Kursus Teknis Prof. (Lvl 5)",
        42: "Kursus Teknis Prof.", 43: "Pendidikan Tinggi - Magister (Siklus 2)"
    },
    "Father's qualification (Pendidikan Ayah)": {
        1: "Pendidikan Menengah (Kelas 12)", 2: "Sarjana Muda (Bachelor)", 3: "Sarjana (Degree)",
        4: "Magister (Master)", 5: "Doktor (PhD)", 6: "Frekuensi Pendidikan Tinggi",
        9: "Kelas 12 (Tidak Tamat)", 10: "Kelas 11 (Tidak Tamat)", 11: "Kelas 7 (Lama)",
        12: "Lainnya - Kelas 11", 14: "Kelas 10", 18: "Kursus Perdagangan Umum",
        19: "Pendidikan Dasar Siklus 3", 26: "Kelas 7", 27: "Kelas 6 (Siklus 2)",
        29: "Kelas 9 (Siklus 3)", 30: "Kursus Lanjutan (Siklus 3)", 34: "Tidak Ada",
        35: "Tidak Bisa Baca Tulis", 36: "Bisa Baca Tanpa Sekolah", 37: "Pendidikan Dasar Siklus 1",
        38: "Pendidikan Dasar Siklus 2", 39: "Kursus Spesialisasi Teknologi", 
        40: "Pendidikan Tinggi - Sarjana (Siklus 1)", 41: "Kursus Teknis Prof. (Lvl 5)",
        42: "Kursus Teknis Prof.", 43: "Pendidikan Tinggi - Magister (Siklus 2)"
    },
    "Mother's occupation (Pekerjaan Ibu)": {
        0: "Mahasiswa", 1: "Manajer/Direktur/Legislatif", 2: "Spesialis Intelektual & Ilmiah",
        3: "Teknisi & Profesi Tingkat Menengah", 4: "Staf Administrasi", 5: "Jasa, Keamanan & Penjual",
        6: "Petani & Pekerja Terampil Pertanian", 7: "Pekerja Terampil Industri/Konstruksi",
        8: "Operator Instalasi & Mesin", 9: "Pekerja Tidak Terampil (Buruh)", 10: "Angkatan Bersenjata",
        90: "Lainnya", 99: "Tidak Diketahui",
        # Detail (ISCO 2-digit)
        11: "Pejabat Legislatif & Senior", 12: "Manajer Admin & Komersial", 13: "Manajer Produksi & Jasa", 14: "Manajer Hotel/Restoran",
        21: "Profesional Sains/Teknik", 22: "Profesional Kesehatan", 23: "Pengajar/Guru/Dosen", 24: "Profesional Bisnis/Hukum", 25: "Profesional TIK", 26: "Profesional Hukum/Budaya",
        27: "Profesional Sains & Teknik Lainnya", 28: "Profesional Kesehatan Lainnya", 29: "Profesional Lainnya",
        30: "Teknisi Umum", 31: "Teknisi Sains & Teknik", 32: "Asisten Profesional Kesehatan", 33: "Asisten Profesional Bisnis", 34: "Asisten Profesional Hukum", 35: "Teknisi TIK",
        36: "Pekerja Layanan Masyarakat", 37: "Pekerja Seni & Budaya", 38: "Teknisi Lainnya", 39: "Teknisi Lainnya",
        40: "Staf Kantor Umum", 41: "Sekretaris/Admin", 42: "Resepsionis/CS", 43: "Pencatat Keuangan/Material", 44: "Staf Pendukung Lain", 45: "Staf Data Entry", 46: "Staf Admin Lainnya",
        51: "Jasa Perorangan (Salon/Travel)", 52: "Pramuniaga/Sales", 53: "Perawat Perorangan", 54: "Satpam/Keamanan"
    },
    "Father's occupation (Pekerjaan Ayah)": {
        0: "Mahasiswa", 1: "Manajer/Direktur/Legislatif", 2: "Spesialis Intelektual & Ilmiah",
        3: "Teknisi & Profesi Tingkat Menengah", 4: "Staf Administrasi", 5: "Jasa, Keamanan & Penjual",
        6: "Petani & Pekerja Terampil Pertanian", 7: "Pekerja Terampil Industri/Konstruksi",
        8: "Operator Instalasi & Mesin", 9: "Pekerja Tidak Terampil (Buruh)", 10: "Angkatan Bersenjata",
        90: "Lainnya", 99: "Tidak Diketahui",
        # Detail (ISCO 2-digit)
        11: "Pejabat Legislatif & Senior", 12: "Manajer Admin & Komersial", 13: "Manajer Produksi & Jasa", 14: "Manajer Hotel/Restoran",
        21: "Profesional Sains/Teknik", 22: "Profesional Kesehatan", 23: "Pengajar/Guru/Dosen", 24: "Profesional Bisnis/Hukum", 25: "Profesional TIK", 26: "Profesional Hukum/Budaya",
        27: "Profesional Sains & Teknik Lainnya", 28: "Profesional Kesehatan Lainnya", 29: "Profesional Lainnya",
        30: "Teknisi Umum", 31: "Teknisi Sains & Teknik", 32: "Asisten Profesional Kesehatan", 33: "Asisten Profesional Bisnis", 34: "Asisten Profesional Hukum", 35: "Teknisi TIK",
        36: "Pekerja Layanan Masyarakat", 37: "Pekerja Seni & Budaya", 38: "Teknisi Lainnya", 39: "Teknisi Lainnya",
        40: "Staf Kantor Umum", 41: "Sekretaris/Admin", 42: "Resepsionis/CS", 43: "Pencatat Keuangan/Material", 44: "Staf Pendukung Lain", 45: "Staf Data Entry", 46: "Staf Admin Lainnya",
        51: "Jasa Perorangan (Salon/Travel)", 52: "Pramuniaga/Sales", 53: "Perawat Perorangan", 54: "Satpam/Keamanan"
    }
}

# Helper untuk memetakan nama kolom asli ke nama yg ada di dictionary (jika beda sedikit)
def get_dict_mapping(col_name):
    # Cari kunci dictionary yang mengandung nama kolom
    for key in DATA_DICTIONARY.keys():
        if col_name in key: # Misal "Course" ada di "Course (Jurusan)"
            return DATA_DICTIONARY[key]
    return None

# --- Logic Class (Modified for Streamlit State) ---
# ## Dokumentasi: Kelas Manajer Aplikasi
# Kelas ini mengatur logika utama aplikasi dan menyimpan data di 'st.session_state'
# agar tidak hilang saat pengguna berinteraksi dengan widget.
class ChurnManager:
    def __init__(self):
        # ## Dokumentasi: Inisialisasi State
        # Menyiapkan variabel di session state jika belum ada.
        # df: dataframe utama, models: menyimpan model yang sudah dilatih, results: menyimpan hasil evaluasi.
        if 'df' not in st.session_state:
            st.session_state['df'] = None
        if 'df_processed' not in st.session_state:
            st.session_state['df_processed'] = None
        if 'models' not in st.session_state:
            st.session_state['models'] = {}
        if 'results' not in st.session_state:
            st.session_state['results'] = {}
        if 'le_target' not in st.session_state:
            st.session_state['le_target'] = None
        if 'best_params' not in st.session_state:
            st.session_state['best_params'] = {}

    def load_data(self, file_path_or_buffer):
        # ## Dokumentasi: Fungsi Memuat Data
        # Membaca file CSV yang diupload atau default, lalu menyimpannya ke st.session_state['df'].
        try:
            # Try loading, handling potential separator issues if any, though standard csv is expected
            df = pd.read_csv(file_path_or_buffer)
            # Remove duplicated spaces in column names if any
            df.columns = df.columns.str.strip()
            
            st.session_state['df'] = df
            st.session_state['df_raw'] = df.copy() # Keep a raw copy
            return True, "Data successfully loaded! (Data berhasil dimuat)"
        except Exception as e:
            return False, str(e)

    def preprocess(self):
        # ## Dokumentasi: Fungsi Preprocessing
        # Tahapan persiapan data:
        # 1. Encoding Target: Mengubah label teks (Dropout/Graduate) menjadi angka (0/1/2).
        # 2. Scaling: Menstandarisasi fitur numerik menggunakan StandardScaler.
        # 3. Splitting: Membagi data menjadi Training (80%) dan Testing (20%).
        if st.session_state['df'] is None:
            return False, "Tidak ada data untuk diproses."
        
        try:
            df = st.session_state['df'].copy()
            
            # --- 1. Target Encoding (Multiclass: Dropout, Enrolled, Graduate) ---
            if 'Target' not in df.columns:
                 return False, "Kolom 'Target' tidak ditemukan di dataset."
            
            le = LabelEncoder()
            df['Target_Encoded'] = le.fit_transform(df['Target'])
            st.session_state['le_target'] = le
            st.session_state['target_classes'] = le.classes_
            
            # --- 2. Feature Selection & X/y ---
            # Drop Target string and Encoded target from X
            X = df.drop(columns=['Target', 'Target_Encoded'])
            y = df['Target_Encoded']
            
            # --- 3. Scaling ---
            # All features in this dataset are either numeric or encoded categorical codes.
            # KNN requires scaling. XGBoost is robust but scaling doesn't hurt.
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Save to state
            st.session_state['X'] = X_scaled
            st.session_state['y'] = y
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = X.columns.tolist()
            
            # --- 4. Split ---
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            st.session_state['split'] = (X_train, X_test, y_train, y_test)
            st.session_state['df_processed'] = X_scaled # For display
            
            return True, "Preprocessing Selesai: Target Encoded -> Fitur Berskala -> Data Dibagi (Split)"
        except Exception as e:
            return False, f"Error: {e}"

    def train_models(self, use_grid=False):
        # ## Dokumentasi: Fungsi Training Model
        # Melatih dua model: XGBoost dan K-Nearest Neighbors (KNN).
        # Jika use_grid=True, akan mencari parameter terbaik (Hyperparameter Tuning).
        # Model yang sudah dilatih disimpan ke st.session_state['models'].
        if 'split' not in st.session_state:
            return False, "Lakukan preprocessing data terlebih dahulu."
        
        X_train, X_test, y_train, y_test = st.session_state['split']
        
        status_text = st.empty()
        progress = st.progress(0)
        
        # --- XGBoost ---
        status_text.text("Training XGBoost... (Melatih XGBoost)")
        # Check target classes count
        num_classes = len(np.unique(y_train))
        
        if use_grid:
             # Simplified Grid for demo speed
             xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), 
                                {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.3]}, cv=3)
             xgb.fit(X_train, y_train)
             xgb_model = xgb.best_estimator_
             st.session_state['best_params']['XGBoost'] = xgb.best_params_
        else:
             xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                       eval_metric='mlogloss', use_label_encoder=False, random_state=42)
             xgb_model.fit(X_train, y_train)
        
        st.session_state['models']['XGBoost'] = xgb_model
        progress.progress(50)
        
        # --- KNN ---
        status_text.text("Training KNN... (Melatih KNN)")
        if use_grid:
            knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}, cv=3)
            knn.fit(X_train, y_train)
            knn_model = knn.best_estimator_
            st.session_state['best_params']['KNN'] = knn.best_params_
        else:
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(X_train, y_train)
            
        st.session_state['models']['KNN'] = knn_model
        progress.progress(90)
        
        # Evaluate
        self.evaluate(X_test, y_test)
        progress.progress(100)
        status_text.text("Training Complete! (Pelatihan Selesai)")
        return True, "Model Telah Dilatih (XGBoost & KNN)"

    def evaluate(self, X_test, y_test):
        # ## Dokumentasi: Fungsi Evaluasi Model
        # Menguji model terhadap data test (yang tidak dilihat saat training).
        # Menghitung metrik Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix.
        results = {}
        for name, model in st.session_state['models'].items():
            y_pred = model.predict(X_test)
            try:
                y_prob = model.predict_proba(X_test)
            except:
                y_prob = None
                
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted'),
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_prob': y_prob
            }
        st.session_state['results'] = results

manager = ChurnManager()

# --- Sidebar ---
# ## Dokumentasi: Sidebar Navigasi
# Menu navigasi di sebelah kiri untuk berpindah halaman utama.
with st.sidebar:
    st.image("https://img.icons8.com/color/100/000000/student-center.png", width=50) 
    st.title("Navigation")
    page = st.radio("Go to:", ["1. Data Exploration", "2. Preprocessing", "3. Model Training", "4. Results & Comparison", "5. Student Success Prediction"])
    
    st.info("💡 **Tips**: Ikuti langkah-langkah secara berurutan.")
    
    st.markdown("---")
    st.write("**Dataset Status:**")
    if st.session_state['df'] is not None:
        st.success("Loaded")
    else:
        st.error("Not Loaded")

# --- Pages ---

if page == "1. Data Exploration":
    # ## Dokumentasi: Halaman 1 - Eksplorasi Data
    # Menampilkan statistik dasar, preview data, dan distribusi fitur.
    # User bisa melihat grafik sebaran data dan kamus data.
    st.title("📊 Data Exploration (Academic Success)")
    
    uploaded_file = st.file_uploader("Upload CSV", type='csv')
    
    # Auto-load default if exists and nothing uploaded yet
    default_path = "dataset.csv"
    if uploaded_file:
         manager.load_data(uploaded_file)
    elif st.session_state['df'] is None and os.path.exists(default_path):
         if st.button("Load Default Dataset (dataset.csv)"):
             manager.load_data(default_path)
             st.rerun()

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Tabs for View
        view_tab, dict_tab = st.tabs(["Dataset View", "📚 Data Dictionary & Meanings"])
        
        with view_tab:
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            # Data Preview
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Distribution Plots
            st.subheader("Target Distribution")
            if 'Target' in df.columns:
                fig = px.pie(df, names='Target', title='Target Distribution', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Target column not found.")
                
            # Feature Distributions
            st.subheader("Feature Explore")
            selected_feat = st.selectbox("Select Feature to Visualize", [c for c in df.columns if c != 'Target'])
            
            if selected_feat:
                c_chart, c_explain = st.columns([2, 1])
                with c_chart:
                    try:
                        # Use a simple histogram compatible with both numeric and categorical
                        # Attempt to map if in dictionary for better legend
                        plot_df = df.copy()
                        
                        # Find mapping
                        mapping = get_dict_mapping(selected_feat)
                        if mapping:
                            plot_df[selected_feat] = plot_df[selected_feat].map(mapping).fillna(plot_df[selected_feat])
                            
                        fig2 = px.histogram(plot_df, x=selected_feat, color='Target', title=f"{selected_feat} Distribution by Target", barmode='group')
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not plot: {e}")
                
                with c_explain:
                    mapping = get_dict_mapping(selected_feat)
                    if mapping:
                        st.info(f"**Arti Kode {selected_feat}:**")
                        for k, v in mapping.items():
                            st.write(f"- **{k}**: {v}")
                    else:
                        st.write("Tidak ada pemetaan kategori khusus untuk fitur numerik/kontinu ini.")
                    
            # Numeric Stats
            st.write("Numerical Statistics:")
            st.dataframe(df.describe())
            
        with dict_tab:
            st.markdown("### Arti Atribut (Kamus Data)")
            st.markdown("Penjelasan mengenai kode angka yang digunakan pada kolom kategorikal dataset.")
            
            dict_col = st.selectbox("Pilih Atribut", list(DATA_DICTIONARY.keys()))
            if dict_col:
                # Convert dict to df for display
                mapping = DATA_DICTIONARY[dict_col]
                # Force index to be string to avoid weird formatting of code
                map_df = pd.DataFrame( list(mapping.items()), columns=['Kode (Angka)', 'Arti / Deskripsi'] )
                st.table(map_df)

elif page == "2. Preprocessing":
    # ## Dokumentasi: Halaman 2 - Preprocessing
    # Menjelaskan dan menjalankan proses pembersihan data.
    # Menampilkan contoh data sebelum dan sesudah diproses.
    st.title("⚙️ Data Preprocessing")
    
    if st.session_state['df'] is None:
        st.warning("Please load data in step 1.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🛠️ Tahapan Preprocessing")
            st.write("Proses ini mengubah data mentah menjadi format yang siap untuk algoritma Machine Learning.")
            
            with st.expander("1. Encoding Target (Label)", expanded=True):
                st.write("**Tujuan:** Mengubah label teks menjadi angka agar dimengerti model.")
                st.markdown("- `Dropout` ➡️ **0**")
                st.markdown("- `Enrolled` ➡️ **1**")
                st.markdown("- `Graduate` ➡️ **2**")
                
            with st.expander("2. Penskalaan Fitur (Standard Scaler)", expanded=True):
                st.write("**Tujuan:** Menyamakan rentang nilai antar fitur agar fitur dengan nilai besar tidak mendominasi.")
                st.write("**Metode:** $z = (x - mean) / std\_dev$")
                st.write("**Contoh:** Umur (18-50) dan Nilai (0-200) akan memiliki skala sebaran yang sama (sekitar -3 sd 3).")
                
            with st.expander("3. Pembagian Data (Train-Test Split)", expanded=True):
                st.write("**Tujuan:** Memisahkan data untuk belajar (Train) dan data untuk ujian (Test).")
                st.write("**Parameter:**")
                st.markdown("- **Ukuran Data Uji (Test Size):** `0.2` (20% data untuk testing)")
                st.markdown("- **Stratifikasi:** `Ya` (Proporsi label pada train dan test akan sama)")
                st.markdown("- **Random State:** `42` (Agar hasil konsisten)")
        
        with c2:
            if st.button("🚀 Jalankan Preprocessing", type="primary"):
                success, msg = manager.preprocess()
                if success:
                    st.success(msg)
                    st.session_state['processed'] = True
                    st.rerun()
                else:
                    st.error(msg)
                    
        if st.session_state.get('df_processed') is not None:
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Target Asli")
                st.write(st.session_state['df']['Target'].unique())
            with col_r:
                st.subheader("Target Ter-Encode")
                st.write(st.session_state['le_target'].classes_)
                
            st.subheader("Contoh Data Hasil Proses (Terskala)")
            st.dataframe(st.session_state['df_processed'].iloc[:5, :5], use_container_width=True)

elif page == "3. Model Training":
    # ## Dokumentasi: Halaman 3 - Training Model
    # Tempat user mengkonfigurasi (GridSearch) dan memulai proses pelatihan model.
    # Menampilkan parameter yang digunakan dan penjelasan singkat algoritma.
    st.title("🧠 Model Training")
    
    if 'split' not in st.session_state:
        st.warning("Please run Preprocessing first.")
    else:
        st.write("Di sini kita akan melatih dua algoritma: **XGBoost** dan **K-Nearest Neighbors (KNN)** untuk membandingkan kinerjanya.")
        
        col_param, col_desc = st.columns(2)
        
        with col_param:
            st.subheader("⚙️ Konfigurasi Parameter")
            use_grid = st.checkbox("Enable Hyperparameter Tuning (GridSearchCV)", help="Mencoba berbagai kombinasi parameter. Proses akan lebih lama.")
            
            st.markdown("**Parameter Default (Tanpa Tuning):**")
            with st.expander("Parameter XGBoost", expanded=True):
                st.markdown("- **n_estimators:** `100` (Jumlah pohon keputusan)")
                st.markdown("- **max_depth:** `6` (Kedalaman maksimal pohon)")
                st.markdown("- **learning_rate:** `0.1` (Kecepatan belajar)")
                st.markdown("- **objective:** `multi:softprob` (Multiclass classification)")
            
            with st.expander("Parameter KNN", expanded=True):
                st.markdown("- **n_neighbors (k):** `5` (Jumlah tetangga terdekat yang dicek)")
                st.markdown("- **metric:** `minkowski` (Metode pengukuran jarak)")
        
        with col_desc:
            st.subheader("📖 Penjelasan Algoritma")
            st.info("""
            **XGBoost (Extreme Gradient Boosting)**
            
            Algoritma berbasis *ensemble* (kumpulan) pohon keputusan. Ia bekerja dengan membuat pohon secara berurutan, di mana setiap pohon baru berusaha memperbaiki kesalahan prediksi dari pohon sebelumnya. Sangat kuat untuk data terstruktur.
            """)
            
            st.success("""
            **KNN (K-Nearest Neighbors)**
            
            Algoritma yang sederhana namun efektif. Ia mengklasifikasikan data baru berdasarkan kemiripan (jarak) dengan data tetangga terdekatnya. Jika tetangganya mayoritas 'Lulus', maka data baru diprediksi 'Lulus'.
            """)

        st.divider()
        if st.button("Start Training", type="primary"):
            success, msg = manager.train_models(use_grid)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        if st.session_state['models']:
            st.success("Models Trained! Go to Results.")
            
            if st.session_state['best_params']:
                st.info("🔎 **Hasil Tuning Hyperparameter (GridSearchCV):**")
                for model_name, params in st.session_state['best_params'].items():
                    st.write(f"**{model_name} Best Params:** `{params}`")

elif page == "4. Results & Comparison":
    # ## Dokumentasi: Halaman 4 - Hasil & Perbandingan
    # Menampilkan tabel metrik performa (Akurasi dll) dan grafik visual.
    # Menyediakan "Deep Dive" untuk melihat Confusion Matrix dan Feature Importance.
    st.title("🏆 Results & Comparison")
    
    if not st.session_state['results']:
        st.warning("Train models first!")
    else:
        results = st.session_state['results']
        
        # 1. Metrics Table
        st.subheader("Model Performance (Weighted Avg)")
        st.write("Nilai mendekati 1.0 berarti performa semakin baik.")
        metrics_data = []
        for model, res in results.items():
            metrics_data.append({
                'Model': model,
                'Accuracy': res['Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1-Score': res['F1']
            })
        metrics_df = pd.DataFrame(metrics_data).set_index('Model')
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='green').format("{:.4f}"), use_container_width=True)
        
        # 2. Side-by-side Charts
        st.subheader("Visual Comparison")
        plot_data = pd.melt(metrics_df.reset_index(), id_vars='Model', var_name='Metric', value_name='Score')
        fig_bar = px.bar(plot_data, x='Metric', y='Score', color='Model', barmode='group', title='Metric Comparison')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 3. Deep Dive
        st.subheader("Deep Dive Analysis")
        tab1, tab2 = st.tabs(["Confusion Matrix", "Feature Importance"])
        
        classes = st.session_state['le_target'].classes_
        
        with tab1:
            st.write("Confusion Matrix menunjukkan detail kesalahan prediksi (misal: Harusnya 'Graduate' tapi diprediksi 'Dropout').")
            c1, c2 = st.columns(2)
            for i, (name, res) in enumerate(results.items()):
                with (c1 if i % 2 == 0 else c2):
                    st.write(f"**{name}**")
                    cm = res['Confusion Matrix']
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                       x=classes, y=classes,
                                       labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig_cm, use_container_width=True)

        with tab2:
            st.write("Fitur apa yang paling mempengaruhi keputusan model?")
            if 'XGBoost' in st.session_state['models']:
                xgb = st.session_state['models']['XGBoost']
                imps = xgb.feature_importances_
                feats = st.session_state['feature_names']
                feat_df = pd.DataFrame({'Feature': feats, 'Importance': imps}).sort_values(by='Importance', ascending=False).head(15)
                
                fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top 15 Predictors (XGBoost)")
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("XGBoost belum dilatih.")

elif page == "5. Student Success Prediction":
    # ## Dokumentasi: Halaman 5 - Prediksi & Insight
    # Simulasi penggunaan model untuk memprediksi data siswa baru secara acak.
    # Menampilkan hasil prediksi dari kedua model berdampingan dan data profil siswa.
    st.title("🎓 Student Success Prediction & Insight")
    
    if 'models' not in st.session_state or ('XGBoost' not in st.session_state['models'] and 'KNN' not in st.session_state['models']):
         st.warning("Please train models first.")
    else:
        st.markdown("Halaman ini mensimulasikan prediksi untuk data mahasiswa tes yang belum pernah dilihat model.")
        
        # Initialize session state for prediction if not exists
        if 'pred_data' not in st.session_state:
            st.session_state['pred_data'] = None

        if st.button("🎲 Pick Random Test Student"):
            X_test_scaled = st.session_state['split'][1]
            y_test = st.session_state['split'][3]
            
            # Pick random index
            random_idx = np.random.randint(0, len(X_test_scaled))
            # Keep as DataFrame to preserve column names
            student_data_scaled = X_test_scaled.iloc[[random_idx]]
            actual_label_code = y_test.iloc[random_idx]
            
            # Save RAW data (not prediction yet) to session state
            st.session_state['pred_data'] = {
                'scaled': student_data_scaled,
                'actual_code': actual_label_code
            }

        # Check if we have data to display
        if st.session_state['pred_data'] is not None:
            data = st.session_state['pred_data']
            student_data_scaled = data['scaled']
            actual_label_code = data['actual_code']
            
            le = st.session_state['le_target']
            actual_label = le.inverse_transform([actual_label_code])[0]
            
            st.divider()
            st.subheader(f"Data Mahasiswa (Actual: {actual_label})")
            
            # --- Results Side-by-Side ---
            c_xgb, c_knn = st.columns(2)
            
            # Logic to display result
            def display_prediction(model_name, col):
                if model_name in st.session_state['models']:
                    model = st.session_state['models'][model_name]
                    with col:
                        st.markdown(f"### {model_name}")
                        pred_code = model.predict(student_data_scaled)[0]
                        try:
                            pred_prob = model.predict_proba(student_data_scaled)[0]
                            confidence = max(pred_prob)
                        except:
                            pred_prob = None
                            confidence = 0.0
                            
                        pred_label = le.inverse_transform([pred_code])[0]
                        
                        if pred_label == actual_label:
                            st.success(f"✅ **{pred_label}**")
                        else:
                            st.error(f"⚠️ **{pred_label}**")
                        
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        if pred_prob is not None:
                            prob_df = pd.DataFrame([pred_prob], columns=le.classes_).T
                            prob_df.columns = ['Prob']
                            st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)

            # Show Both
            display_prediction("XGBoost", c_xgb)
            display_prediction("KNN", c_knn)
            
            st.divider()
            
            # --- Human Readable Conversion ---
            # 1. Inverse Scaling
            scaler = st.session_state['scaler']
            student_data_original_np = scaler.inverse_transform(student_data_scaled)
            student_data_readable = pd.DataFrame(student_data_original_np, columns=student_data_scaled.columns)
            
            # 2. Decode Categorical Features
            for col in student_data_readable.columns:
                mapping = get_dict_mapping(col)
                if mapping:
                    val_float = student_data_readable[col].values[0]
                    val_int = int(round(val_float))
                    text_value = mapping.get(val_int, val_int)
                    student_data_readable[col] = str(text_value) + f" ({val_int})"
                else:
                    student_data_readable[col] = student_data_readable[col].round(2)

            col_profile, col_raw = st.columns([2, 1])
            with col_profile:
                st.subheader("Profil Mahasiswa (Data Asli)")
                st.info("Data ini telah dikembalikan ke format yang mudah dibaca manusia.")
                st.dataframe(student_data_readable.T, use_container_width=True, height=500)
            
            with col_raw:
                 st.subheader("Data Scaled")
                 st.write("Data input untuk model:")
                 st.dataframe(student_data_scaled.T, use_container_width=True)


        # --- Data Dictionary Expander ---
        st.divider()
        with st.expander("📚 Data Dictionary (Check Codes Here)"):
            dict_col = st.selectbox("Check Attribute Codes:", list(DATA_DICTIONARY.keys()), key='dict_pred')
            mapping = DATA_DICTIONARY[dict_col]
            map_df = pd.DataFrame( list(mapping.items()), columns=['Code', 'Meaning/Description'] )
            st.table(map_df)
