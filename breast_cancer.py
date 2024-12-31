import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input  # atau sesuai kebutuhan
from PIL import Image

# -------------------------------------------
# 1. Load model (dengan caching agar tidak re-load setiap ada interaksi)
# -------------------------------------------
@st.cache_resource
def load_breast_cancer_model():
    model = load_model('breast_cancer.h5')
    return model

model = load_breast_cancer_model()

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
    
    return predicted_label, probability

# -------------------------------------------
# 4. Streamlit App
# -------------------------------------------
st.title("Breast Cancer Detection App")
st.write("Aplikasi ini menggunakan model CNN untuk memdeteksi apakah gambar citra Ultrasound dari payudara itu terkena kanker atau normal. Silakan upload gambar MRI di bawah ini untuk diprediksi.")

# Upload file
uploaded_file = st.file_uploader("Upload gambar di sini", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Baca file sebagai PIL Image
    img_pil = Image.open(uploaded_file)
    
    # Tampilkan gambar yang diupload
    st.image(img_pil, caption="Gambar yang di-upload", use_column_width=True)
    
    # Tombol prediksi
    if st.button("Prediksi"):
        with st.spinner("Memproses..."):
            predicted_label, probability = predict_breast_cancer(img_pil)
        
        # Tampilkan hasil
        st.success(f"Hasil Prediksi: **{predicted_label.capitalize()}**")
        st.info(f"Probabilitas: **{probability:.2f}%**")
