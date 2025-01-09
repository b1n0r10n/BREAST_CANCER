import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input  # atau sesuai kebutuhan
from PIL import Image
import pandas as pd  # Untuk visualisasi dan download data

# -------------------------------------------
# 1. Load model (dengan caching agar tidak re-load setiap ada interaksi)
# -------------------------------------------
@st.cache_resource
def load_breast_cancer_model():
    # ID file Google Drive untuk model
    file_id = "1KYlavpAChrn_HhL5DGnM_o9PTKFeqfK1"  # Ganti dengan ID file model Anda di Google Drive
    model_path = "breast_cancer.h5"
    
    # Cek apakah model sudah ada di lokal, jika tidak, unduh dari Google Drive
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model dari Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                st.success("Model berhasil diunduh.")
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                return None

    # Memuat model menggunakan TensorFlow
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_breast_cancer_model()

# Pastikan model berhasil dimuat sebelum melanjutkan
if model is None:
    st.stop()

# -------------------------------------------
# 2. Siapkan dictionary label
# -------------------------------------------
class_labels = {
    0: 'benign',
    1: 'malignant',
    2: 'normal'
}

# -------------------------------------------
# 3. Fungsi prediksi
# -------------------------------------------
def predict_breast_cancer(img_pil):
    """
    Menerima input berupa PIL Image, melakukan preprocessing,
    lalu mengembalikan label dan probabilitas kelas terprediksi.
    """
    try:
        img_height, img_width = 224, 224  # sesuaikan dengan input layer model Anda
        
        # Resize gambar
        img_pil = img_pil.resize((img_height, img_width))
        
        # Konversi ke array
        img_array = image.img_to_array(img_pil)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocessing (misal VGG16, bisa disesuaikan)
        img_array = preprocess_input(img_array)
        
        # Prediksi
        predictions = model.predict(img_array)
        
        # Ambil indeks kelas tertinggi
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        
        # Ambil probabilitas kelas terpilih (hanya satu kelas teratas)
        probability = predictions[0][predicted_class] * 100
        
        return predicted_label, probability, predictions
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None, None, None

# -------------------------------------------
# 4. Streamlit App
# -------------------------------------------

# -------------------------------------------
# 4.1 Tambahkan Navigasi ke Website Utama
# -------------------------------------------
st.sidebar.title("Navigasi")
main_website_url = "https://k11-cnn-detection.vercel.app/"  # Ganti dengan URL website utama Anda
st.sidebar.markdown(f"[🔙 Kembali ke Website Utama]({main_website_url})")

# -------------------------------------------
# 4.2 Judul dan Deskripsi Aplikasi
# -------------------------------------------
st.title("Breast Cancer Detection App")
st.write("""
Aplikasi ini menggunakan model CNN untuk mendeteksi apakah gambar Ultrasound termasuk dalam kategori 
**Benign**, **Malignant**, atau **Normal**. Silakan upload gambar Ultrasound di bawah ini untuk diprediksi.
""")

# -------------------------------------------
# 4.3 Widget File Uploader
# -------------------------------------------
uploaded_file = st.file_uploader("Upload Gambar Ultrasound", type=["png", "jpg", "jpeg"])

# -------------------------------------------
# 4.4 Tampilkan Gambar, Prediksi, dan Fitur Tambahan
# -------------------------------------------
if uploaded_file is not None:
    try:
        # Baca file sebagai PIL Image
        img_pil = Image.open(uploaded_file)
        
        # Tampilkan gambar yang di-upload
        st.image(img_pil, caption="Gambar yang di-upload", use_column_width=True)
        
        # Tombol prediksi
        if st.button("Prediksi"):
            with st.spinner("Memproses..."):
                predicted_label, probability, predictions = predict_breast_cancer(img_pil)
            
            if predicted_label is not None:
                # Tampilkan hasil prediksi
                st.success(f"Hasil Prediksi: **{predicted_label.capitalize()}**")
                st.info(f"Probabilitas: **{probability:.2f}%**")
                
                # -------------------------------------------
                # 4.5 Menambahkan Visualisasi Probabilitas
                # -------------------------------------------
                st.write("Probabilitas untuk setiap kelas:")
                prob_values = predictions[0] * 100
                prob_labels = [class_labels[i] for i in range(len(class_labels))]
                
                # Membuat DataFrame untuk visualisasi
                df_probs = pd.DataFrame({
                    'Kelas': prob_labels,
                    'Probabilitas (%)': prob_values
                }).set_index('Kelas')
                
                # Menampilkan grafik batang
                st.bar_chart(df_probs)
                
                # -------------------------------------------
                # 4.6 Menambahkan Opsi Download Hasil Prediksi
                # -------------------------------------------
                st.write("Anda dapat mendownload hasil prediksi dalam bentuk file CSV.")
                
                # Membuat DataFrame untuk hasil prediksi
                df_result = pd.DataFrame({
                    'Label': [predicted_label],
                    'Probabilitas (%)': [probability]
                })
                
                # Tombol download
                st.download_button(
                    label="Download Hasil Prediksi",
                    data=df_result.to_csv(index=False),
                    file_name='hasil_prediksi.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
