#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAW 图像读取：rawpy / Pillow 后端，统一返回 QImage / QPixmap。"""

import io
import numpy as np

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    from PIL import Image as PILImage, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from qt_compat import QImage, QPixmap, _FMT_RGB888, _KEEP_AR, _SMOOTH_XFORM
from utils import ndarray_to_qimage


# ── PIL → QImage ──────────────────────────────────────────────────────

def pil_to_qimage(img) -> QImage:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return QImage(
        img.tobytes(), img.width, img.height, img.width * 3, _FMT_RGB888
    ).copy()


# ── RAW 读取器 ─────────────────────────────────────────────────────────

class RAWReader:

    @staticmethod
    def read_qimage(path):
        if not HAS_RAWPY:
            return None, "rawpy 未安装"
        try:
            with rawpy.imread(path) as raw:
                qi, _ = RAWReader._thumb_qimage(raw)
                if qi and not qi.isNull():
                    return qi, ""
                qi, err = RAWReader._decode_qimage(raw)
                return (qi, "") if qi and not qi.isNull() else (None, err or "解码失败")
        except rawpy.LibRawFileUnsupportedError:
            return RAWReader._pillow_qimage(path)
        except rawpy.LibRawIOError as e:
            return None, f"读取失败: {e}"
        except Exception as e:
            return None, f"异常: {e}"

    @staticmethod
    def read(path):
        qi, err = RAWReader.read_qimage(path)
        return (QPixmap.fromImage(qi), "") if qi and not qi.isNull() else (None, err)

    @staticmethod
    def _thumb_qimage(raw):
        try:
            t = raw.extract_thumb()
            if t.format == rawpy.ThumbFormat.JPEG:
                data = bytes(t.data)
                if HAS_PIL:
                    try:
                        img = PILImage.open(io.BytesIO(data))
                        return pil_to_qimage(ImageOps.exif_transpose(img)), ""
                    except Exception:
                        pass
                qi = QImage()
                if qi.loadFromData(data) and qi.width() > 0:
                    return qi, ""
            elif t.format == rawpy.ThumbFormat.BITMAP:
                qi = ndarray_to_qimage(np.array(t.data))
                if qi and qi.width() > 0:
                    return qi, ""
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            pass
        except Exception as e:
            print(f"[thumb] {e}")
        return None, ""

    @staticmethod
    def _decode_qimage(raw):
        try:
            rgb = raw.postprocess(
                use_camera_wb=True, half_size=False,
                no_auto_bright=False, output_bps=8)
            qi = ndarray_to_qimage(rgb)
            return (qi, "") if qi and not qi.isNull() else (None, "解码输出为空")
        except Exception as e:
            return None, f"解码失败: {e}"

    @staticmethod
    def _pillow_qimage(path):
        if not HAS_PIL:
            return None, "不支持的格式且 Pillow 未安装"
        try:
            img = PILImage.open(path)
            img = ImageOps.exif_transpose(img)
            return pil_to_qimage(img), ""
        except Exception as e:
            return None, f"Pillow 失败: {e}"

    @staticmethod
    def read_qimage_thumb_and_full(path):
        """评分专用：一次 rawpy.imread 同时返回 (thumb_qi, full_qi, err)。

        • 有嵌入缩略图时 thumb_qi == full_qi（同一对象，零额外解码）。
        • 无缩略图时两者都指向全分辨率解码结果。
        thumb_qi → YOLO BGR 转换；full_qi → Laplacian 评分。
        """
        if not HAS_RAWPY:
            return None, None, "rawpy 未安装"
        try:
            with rawpy.imread(path) as raw:
                thumb_qi, _ = RAWReader._thumb_qimage(raw)
                if thumb_qi and not thumb_qi.isNull():
                    return thumb_qi, thumb_qi, ""
                full_qi, err = RAWReader._decode_qimage(raw)
                if full_qi and not full_qi.isNull():
                    return full_qi, full_qi, ""
                return None, None, err or "解码失败"
        except rawpy.LibRawFileUnsupportedError:
            qi, err = RAWReader._pillow_qimage(path)
            return qi, qi, err
        except rawpy.LibRawIOError as e:
            return None, None, f"读取失败: {e}"
        except Exception as e:
            return None, None, f"异常: {e}"
