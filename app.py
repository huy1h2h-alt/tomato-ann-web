"""Simple mobile-first Streamlit UI for the existing classical ANN model.

Image upload uses the phone's native camera/gallery picker when available.
The live camera option needs HTTPS on mobile browsers (or localhost on PC).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from predict import DEFAULT_MODEL, classify, load_model
from mobile_image import load_mobile_image

APP_VERSION = "3.0.0"

st.set_page_config(
    page_title="Kiểm tra lá cà chua",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# One-column UI, with larger touch targets on mobile devices.
st.markdown(
    """
    <style>
      .block-container {max-width: 580px; padding-top: 1.1rem; padding-bottom: 2rem;}
      #MainMenu, footer {visibility: hidden;}
      div.stButton > button {min-height: 3.1rem; font-size: 1.08rem; font-weight: 650;}
      @media (max-width: 600px) {
        .block-container {padding-left: .85rem; padding-right: .85rem;}
        h1 {font-size: 1.75rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍅 Kiểm tra lá cà chua")

st.caption("Chụp một lá rõ nét để kiểm tra dấu hiệu bệnh.")


@st.cache_resource(show_spinner=False)
def get_model(path: str, modified_ns: int, size: int):
    """Reload only when the trained model file changes."""
    return load_model(path)


model_file = Path(DEFAULT_MODEL)
if not model_file.is_file():
    st.error("Chưa có mô hình. Hãy huấn luyện ANN trước khi mở app.")
    st.code("python train.py --data data/processed", language="bash")
    st.stop()

try:
    artifact = get_model(
        str(model_file), model_file.stat().st_mtime_ns, model_file.stat().st_size
    )
except Exception as exc:
    st.error(f"Không thể mở mô hình ANN: {exc}")
    st.stop()

# Default option stays usable on a phone even when browser camera permission
# is restricted; many mobile browsers offer 'Take Photo' in the file picker.
source = st.radio(
    "Chọn cách lấy ảnh",
    ("📱 Chụp / chọn ảnh", "🎥 Camera trực tiếp"),
    horizontal=True,
)

if source == "📱 Chụp / chọn ảnh":
    uploaded = st.file_uploader(
        "Nhấn để mở camera hoặc chọn ảnh trong điện thoại",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="mobile_photo",
    )
else:
    st.caption("Camera trực tiếp trên điện thoại cần liên kết HTTPS và quyền truy cập camera.")
    uploaded = st.camera_input("Chụp lá cà chua", key="live_camera")

if uploaded is not None:
    try:
        photo = load_mobile_image(uploaded.getvalue())
        st.image(photo, caption="Ảnh cần kiểm tra", use_container_width=True)
        if st.button("🔍 KIỂM TRA LÁ", type="primary", use_container_width=True):
            with st.spinner("Đang kiểm tra..."):
                result = classify(photo, artifact)

            decision = result["decision"]
            if decision == "healthy":
                st.success("🟢 KHÔNG PHÁT HIỆN DẤU HIỆU BỆNH")
                st.caption("Mô hình nghiêng về lá khỏe. Hãy tiếp tục theo dõi lá thực tế.")
            elif decision == "diseased":
                st.error("🔴 LÁ CÓ DẤU HIỆU BỆNH")
                st.caption("Mô hình nghiêng về lá bệnh; chưa xác định tên bệnh cụ thể.")
            elif decision == "uncertain":
                st.warning("🟡 CHƯA THỂ KẾT LUẬN")
                st.caption("Hãy chụp thêm một ảnh rõ nét, đủ sáng để kiểm tra lại.")
            else:
                st.warning("⚪ ẢNH CHƯA PHÙ HỢP")
                st.caption("Hãy chụp một lá riêng biệt trên nền đơn giản.")

            if result["warnings"] and decision in ("invalid", "uncertain"):
                st.caption("Gợi ý: " + " ".join(result["warnings"]))

    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Không kiểm tra được ảnh. Vui lòng thử lại hoặc xem lỗi trong Terminal.")
        # Full exception is printed only to the server log, not to the mobile UI.
        import logging

        logging.getLogger(__name__).exception("Mobile ANN inference failed: %s", exc)

st.divider()
st.caption("Công cụ tham khảo cho lá cà chua; kết quả ANN không thay thế việc kiểm tra thực tế.")
