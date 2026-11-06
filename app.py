"""Streamlit app tối giản: Chỉ hiển thị kết quả lá có bệnh hay không."""
from pathlib import Path
import io

import streamlit as st
from PIL import Image, ImageOps
from predict import load_model, classify

st.set_page_config(page_title='Chẩn đoán lá cà chua', page_icon='🍅', layout='centered')

st.markdown('''
<style>
.block-container { max-width: 500px; padding-top: 1.5rem; }
.result-box {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: bold;
    margin-top: 1.2rem;
}
.diseased { background-color: #ffebe9; color: #cf222e; border: 2px solid #ff8182; }
.healthy { background-color: #dafbe1; color: #1a7f37; border: 2px solid #4ac26b; }
</style>
''', unsafe_allow_html=True)

st.title("🍅 Kiểm Tra Bệnh Lá Cà Chua")

model_file = Path(__file__).resolve().parent / 'models' / 'tomato_ann.joblib'

@st.cache_resource(show_spinner=False)
def get_model(path, mtime):
    return load_model(path)

if not model_file.exists():
    st.error("Chưa tìm thấy tệp mô hình trong thư mục `models/`.")
    st.stop()

try:
    artifact = get_model(str(model_file), model_file.stat().st_mtime_ns)
except Exception as e:
    st.error(f"Lỗi nạp mô hình: {e}")
    st.stop()

mode = st.radio("Chọn phương thức:", ["📷 Chụp ảnh", "📁 Tải ảnh từ máy"], horizontal=True)

image_file = None
if mode == "📷 Chụp ảnh":
    image_file = st.camera_input("Chụp lá cà chua")
else:
    image_file = st.file_uploader("Tải file ảnh lá", type=['jpg', 'jpeg', 'png', 'webp'])

if image_file is not None:
    try:
        raw_img = Image.open(io.BytesIO(image_file.getvalue()))
        raw_img = ImageOps.exif_transpose(raw_img)
        
        # Chuyển nền trong suốt của PNG thành nền trắng
        if raw_img.mode in ('RGBA', 'LA') or (raw_img.mode == 'P' and 'transparency' in raw_img.info):
            rgba = raw_img.convert('RGBA')
            photo = Image.new('RGB', rgba.size, (255, 255, 255))
            photo.paste(rgba, mask=rgba.split()[3])
        else:
            photo = raw_img.convert('RGB')

        with st.spinner('Đang kiểm tra...'):
            res = classify(photo, artifact)
            
        st.image(photo, use_container_width=True)
        
        if res['diseased']:
            st.markdown('<div class="result-box diseased">🔴 LÁ CÓ DẤU HIỆU BỆNH</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box healthy">🟢 LÁ KHỎE MẠNH</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Lỗi đọc ảnh: {e}")