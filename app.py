"""Streamlit app tối giản: Chẩn đoán lá cà chua khỏe mạnh hay bị bệnh."""
from pathlib import Path
import io

import streamlit as st
from PIL import Image, ImageOps
from predict import load_model, classify

# Cấu hình giao diện gọn nhẹ
st.set_page_config(page_title='Chẩn đoán lá cà chua', page_icon='🍅', layout='centered')

# Tùy chỉnh hiển thị khung kết quả rõ ràng
st.markdown('''
<style>
.block-container { max-width: 550px; padding-top: 1.5rem; }
.result-box {
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: bold;
    margin-top: 1rem;
}
.diseased { background-color: #ffebe9; color: #cf222e; border: 2px solid #ff8182; }
.healthy { background-color: #dafbe1; color: #1a7f37; border: 2px solid #4ac26b; }
</style>
''', unsafe_allow_html=True)

st.title("🍅 Kiểm Tra Bệnh Lá Cà Chua")

# Nạp mô hình
model_file = Path(__file__).resolve().parent / 'models' / 'tomato_ann.joblib'

@st.cache_resource(show_spinner=False)
def get_model(path, mtime):
    return load_model(path)

if not model_file.exists():
    st.error("Chưa tìm thấy mô hình `models/tomato_ann.joblib`.")
    st.stop()

artifact = get_model(str(model_file), model_file.stat().st_mtime_ns)

# Chọn nguồn ảnh đơn giản
mode = st.radio("Chọn phương thức:", ["📷 Chụp ảnh", "📁 Tải ảnh từ máy"], horizontal=True)

image_file = None
if mode == "📷 Chụp ảnh":
    image_file = st.camera_input("Chụp lá cà chua")
else:
    image_file = st.file_uploader("Tải file ảnh lá", type=['jpg', 'jpeg', 'png', 'webp'])

# Hiển thị duy nhất kết quả
if image_file is not None:
    try:
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(image_file.getvalue()))).convert('RGB')
        
        with st.spinner('Đang kiểm tra...'):
            res = classify(photo, artifact)
            
        st.image(photo, use_container_width=True)
        
        if res['diseased']:
            st.markdown('<div class="result-box diseased">🔴 LÁ CÓ DẤU HIỆU BỆNH</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box healthy">🟢 LÁ KHỎE MẠNH</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Lỗi đọc ảnh: {e}")