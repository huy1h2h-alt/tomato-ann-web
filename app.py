"""Streamlit app: phone/PC camera input and upload for disease vs healthy."""
from __future__ import annotations

from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from predict import load_model, classify

st.set_page_config(page_title='Tomato Leaf · ANN Detector', page_icon='🍅',
                   layout='wide', initial_sidebar_state='collapsed')
st.markdown('''
<style>
.stApp { background: #fafcf9; }
.block-container { max-width: 1060px; padding-top: 2rem; }
h1, h2, h3 { color: #193e2a; }
[data-testid="stMetricValue"] { color: #193e2a; }
.hero {padding: 1.35rem 1.5rem; border-radius: 18px;
       background: #eaf5e8; border: 1px solid #d0e6d1; margin-bottom: 1.2rem; }
.hero h1 {margin:0; font-size: 2rem; }
.hero p {margin:.55rem 0 0 0; color:#3d654c; font-size:1.02rem; }
.note {color:#52705c; font-size:.88rem; }
.status {padding:1rem 1.2rem; border-radius: 14px; margin: .7rem 0; font-weight: 700; font-size:1.16rem;}
.good { background:#e5f5e8; color:#176337; border:1px solid #b7dec0; }
.bad { background:#fff0e8; color:#a14124; border:1px solid #f3c2a7; }
</style>
''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><h1>🍅 Tomato Leaf Detector</h1>
<p>Phân loại lá cà chua: <b>có dấu hiệu bệnh</b> hoặc <b>không phát hiện bệnh</b> bằng ANN truyền thống.</p>
</div>''', unsafe_allow_html=True)

model_file = Path(__file__).resolve().parent / 'models' / 'tomato_ann.joblib'
if not model_file.exists():
    st.warning('Chưa tìm thấy mô hình đã huấn luyện. Cần chuẩn bị 2.000 ảnh, chạy train.py trước, sau đó mở lại app.')
    st.code('python download_2000.py\npython train.py\npython -m streamlit run app.py', language='bash')
    st.info('Nếu có ảnh riêng, hãy dùng prepare_local.py thay cho download_2000.py.')
    st.stop()

@st.cache_resource(show_spinner=False)
def cached_model(path, modified_time):
    return load_model(path)

try:
    artifact = cached_model(str(model_file), model_file.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'Không mở được model: {exc}')
    st.stop()

with st.sidebar:
    st.header('Thông tin mô hình')
    st.write('**Phương pháp:** đặc trưng thủ công + StandardScaler + MLP ANN')
    st.write('**Không sử dụng:** CNN, transfer learning, mạng pretrained.')
    st.write(f'**Số đặc trưng:** {artifact["feature_count"]}')
    st.write(f'**Lớp ẩn:** {artifact["architecture"]}')
    st.write(f'**Ngưỡng Validation:** {artifact["threshold"]:.3f}')
    metrics = artifact.get('test_metrics', {})
    if metrics:
        st.write(f'**Accuracy trên 300 ảnh test:** {metrics["accuracy"]:.1%}')
        st.write(f'**Recall lá bệnh:** {metrics["recall_diseased"]:.1%}')
    st.caption('Kết quả test trên PlantVillage không đảm bảo độ chính xác trên ảnh ngoài thực tế.')

st.subheader('1. Chụp hoặc tải ảnh lá')
st.caption('Đặt **một lá cà chua** trên **nền trắng/xám trơn**, ánh sáng đều, không bị che, lá nằm gọn trong khung ảnh.')
mode = st.radio('Nguồn hình ảnh', ['Chụp ảnh bằng camera', 'Tải ảnh lên'], horizontal=True)
if mode == 'Chụp ảnh bằng camera':
    image_file = st.camera_input('Mở camera và chụp lá cà chua', key='tomato_camera')
else:
    image_file = st.file_uploader('Chọn JPG, PNG hoặc WEBP', type=['jpg','jpeg','png','webp'], key='tomato_upload')

if image_file is not None:
    try:
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(image_file.getvalue()))).convert('RGB')
        if photo.width < 80 or photo.height < 80:
            st.error('Ảnh quá nhỏ, hãy dùng ảnh có chiều ngang và dọc từ 80 pixel trở lên.')
            st.stop()
        with st.spinner('Đang tách nền, trích đặc trưng và phân loại...'):
            pred = classify(photo, artifact)
        st.divider()
        st.subheader('2. Kết quả phân loại')
        style = 'bad' if pred['diseased'] else 'good'
        st.markdown(f'<div class="status {style}">{pred["label"]}</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric('Xác suất dấu hiệu bệnh*', f'{pred["prob_disease"]:.1%}')
        col2.metric('Xác suất không bệnh*', f'{pred["prob_healthy"]:.1%}')
        col3.metric('Tỷ lệ diện tích lá trong ảnh', f'{pred["segmentation"].coverage:.0%}')
        st.progress(float(np.clip(pred['prob_disease'], 0, 1)),
                    text=f'Mức dự đoán lá bệnh: {pred["prob_disease"]:.1%}')
        if pred['margin'] < 0.12:
            st.warning('Dự đoán gần ngưỡng phân loại, độ tin cậy quyết định thấp. Hãy thử ảnh rõ hơn.')
        if pred['segmentation'].warning:
            st.warning('Chất lượng ảnh: ' + pred['segmentation'].warning)
        ca, cb, cc = st.columns(3)
        with ca:
            st.image(photo, caption='Ảnh gốc', use_container_width=True)
        with cb:
            st.image(pred['segmentation'].mask, caption='Mặt nạ tách lá (trắng: lá)',
                     clamp=True, use_container_width=True)
        with cc:
            masked = pred['rgb'].copy()
            masked[pred['segmentation'].mask == 0] = (246, 248, 246)
            st.image(masked, caption='Lá sau tách nền', use_container_width=True)
        st.caption('*Đây là điểm xác suất do ANN ước lượng, chưa được hiệu chuẩn như một xác suất y khoa/nông học. '
                   'Ứng dụng chỉ phát hiện dấu hiệu bất thường trên ảnh, không xác định loại bệnh hoặc phương pháp điều trị.')
        if pred['diseased']:
            st.info('Gợi ý: chụp thêm mặt trên/dưới lá, kiểm tra các lá khác, và nhờ người có chuyên môn nông nghiệp xác định bệnh cụ thể.')
        else:
            st.info('Chưa phát hiện dấu hiệu bệnh trong ảnh này. Không loại trừ bệnh giai đoạn sớm hoặc ngoài các bệnh trong dữ liệu huấn luyện.')
    except Exception as exc:
        st.error(f'Không thể xử lý ảnh: {exc}')

st.divider()
st.subheader('3. Mẹo để dự đoán ổn định')
st.markdown('• Chỉ chụp **1 lá cà chua**, tránh chụp cả cây hoặc nhiều lá chồng lên nhau.  \n'
            '• Dùng nền đơn sắc khác màu lá, ánh sáng ban ngày đều, tránh bóng gắt.  \n'
            '• Lá chiếm khoảng **25–75%** khung hình, thấy rõ cả đường viền và các đốm bệnh.  \n'
            '• Nếu ảnh có nhiều lá, nền phức tạp hoặc ánh sáng quá yếu, hãy chụp lại.')
st.caption('Ứng dụng hoạt động trên trình duyệt. Quyền camera được trình duyệt hỏi khi nhấn chụp; '
           'máy tính chạy app cần cấp quyền camera. Để sử dụng trên điện thoại khác thiết bị, '
           'trình duyệt thường yêu cầu kết nối HTTPS.')
