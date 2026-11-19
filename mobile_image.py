"""Validate and resize mobile camera/gallery images before ANN inference.

Converts phone photographs to a bounded RGB image. The ANN feature algorithm
in features.py stays unchanged, so existing trained models remain compatible.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 50_000_000  # Allow common 48 MP smartphone photos.
MAX_PREVIEW_SIDE = 1600
# PIL is global; set the same read limit once rather than changing it during
# each Streamlit request (concurrent sessions might otherwise race).
Image.MAX_IMAGE_PIXELS = max(int(Image.MAX_IMAGE_PIXELS or 0), MAX_SOURCE_PIXELS)


def load_mobile_image(content: bytes) -> Image.Image:
    """Return an EXIF-corrected RGB image with bounded dimensions.

    Raises ValueError on oversized or invalid images; never trusts filename/MIME.
    """
    # features.py sets Pillow's default to 24 MP during import; for this one
    # mobile entry point, raise the decoder safety ceiling to 50 MP before open.
    # A stable global cap (not per-request enable/disable) is thread-safe.
    Image.MAX_IMAGE_PIXELS = max(int(Image.MAX_IMAGE_PIXELS or 0), MAX_SOURCE_PIXELS)
    if not content:
        raise ValueError("Chưa có dữ liệu ảnh. Hãy chọn ảnh khác.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Ảnh lớn hơn 20 MB. Hãy chụp lại với độ phân giải thấp hơn.")

    # The training module restricts PIL images to 24 MP. For the phone input,
    # permit up to 50 MP *only until* we downscale before feature extraction.
    try:
        with Image.open(io.BytesIO(content)) as src:
            width, height = src.size
            if width < 32 or height < 32:
                raise ValueError("Ảnh quá nhỏ. Hãy chụp lại gần lá hơn.")
            if width * height > MAX_SOURCE_PIXELS:
                raise ValueError("Ảnh vượt quá 50 megapixel. Hãy giảm độ phân giải.")

            # JPEG draft decodes at lower resolution when the codec supports it.
            if src.format == "JPEG":
                src.draft("RGB", (MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE))
            photo = ImageOps.exif_transpose(src)
            photo.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE), Image.Resampling.LANCZOS)
            if photo.mode in ("RGBA", "LA") or (
                photo.mode == "P" and "transparency" in photo.info
            ):
                rgba = photo.convert("RGBA")
                white = Image.new("RGB", rgba.size, (255, 255, 255))
                white.paste(rgba, mask=rgba.getchannel("A"))
                return white
            return photo.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Không đọc được ảnh. Hãy dùng JPG, PNG hoặc WebP.") from exc
    except OSError as exc:
        raise ValueError("Ảnh bị lỗi hoặc định dạng chưa được hỗ trợ. Hãy dùng JPG/PNG.") from exc
