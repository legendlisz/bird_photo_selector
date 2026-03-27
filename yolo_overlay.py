#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 全局锁：防止多个线程同时调用 YOLO 推理（PyTorch / CUDA 不支持并发推理）
_YOLO_LOCK = threading.Lock()
YOLO_BIRD_MIN_CONF = 0.35
YOLO_EYE_MIN_CONF  = 0.30


def _load_qt():
    try:
        from PySide6.QtCore import QObject, QThread, Signal
        from PySide6.QtGui import QImage
        return QObject, QThread, Signal, QImage
    except Exception:
        from PyQt5.QtCore import QObject, QThread, pyqtSignal as Signal
        from PyQt5.QtGui import QImage
        return QObject, QThread, Signal, QImage


QObject, QThread, Signal, QImage = _load_qt()


def _load_ultralytics_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception:
        return None


YOLO = _load_ultralytics_yolo()

# 模型缓存
_model_cache = {}

def _get_model(model_path: str):
    """获取模型实例（线程安全单例）"""
    global _model_cache
    model_path = _ensure_file(model_path)
    # 快速路径：已缓存直接返回（GIL 保证 dict 读是原子的）
    if model_path in _model_cache:
        return _model_cache[model_path]
    # 慢速路径：首次加载，加锁防止并发初始化
    with _YOLO_LOCK:
        if model_path not in _model_cache:   # 二次检查
            try:
                _model_cache[model_path] = YOLO(model_path)
            except Exception:
                _model_cache[model_path] = None
    return _model_cache[model_path]


from utils import _qimage_to_bgr


def _downsample_points_xy(xy: np.ndarray, max_points: int = 160) -> List[Tuple[int, int]]:
    if xy is None:
        return []
    try:
        pts = np.asarray(xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return []
        if len(pts) == 0:
            return []
        if len(pts) > max_points:
            step = max(1, len(pts) // max_points)
            pts = pts[::step]
        return [(int(round(x)), int(round(y))) for x, y in pts.tolist()]
    except Exception:
        return []


def _ensure_file(path: str) -> str:
    p = str(Path(path).resolve())
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p


class _OverlayThread(QThread):
    sig_ready = Signal(str, object)

    def __init__(self, path: str, qimage: QImage, bird_model: str, eye_model: str, parent=None):
        super().__init__(parent)
        self._path = str(Path(path).resolve())
        self._qimage = qimage.copy() if qimage is not None else None
        self._bird_model_path = bird_model
        self._eye_model_path = eye_model
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self._cancelled:
            return

        if YOLO is None:
            print('[YOLO] ultralytics 未安装，跳过推理 (pip install ultralytics)')
            self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
            return

        if self._qimage is None or self._qimage.isNull():
            print('[YOLO] qimage 为空，跳过推理')
            self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
            return

        img = _qimage_to_bgr(self._qimage)
        if img is None or img.size == 0:
            print('[YOLO] 图像转换失败，跳过推理')
            self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
            return

        H, W = img.shape[:2]

        bird_boxes: List[Tuple[int, int, int, int, float]] = []
        bird_polys: List[List[Tuple[int, int]]] = []
        eye_boxes: List[Tuple[int, int, int, int, float]] = []

        try:
            bird_model = _get_model(self._bird_model_path)
            if bird_model is None:
                print(f'[YOLO] 鸟身模型加载失败：{self._bird_model_path}')
                self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
                return
        except FileNotFoundError:
            print(f'[YOLO] 鸟身模型文件不存在：{self._bird_model_path}')
            self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
            return
        except Exception as e:
            print(f'[YOLO] 鸟身模型加载异常：{e}')
            self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
            return

        if self._cancelled:
            return

        try:
            with _YOLO_LOCK:
                res = bird_model(img, imgsz=1024, verbose=False)[0]
        except Exception as e:
            print(f'[YOLO] 鸟身推理失败：{e}')
            self.sig_ready.emit(self._path, {"bird_boxes": [], "bird_polys": [], "eye_boxes": []})
            return

        if self._cancelled:
            return

        names = getattr(bird_model, "names", None)
        # 兼容专用鸟类模型（单类或多类细分）：收集所有与鸟相关的 class id。
        # COCO 通用模型中 bird=14；专用模型通常 class 0 = 'bird'，
        # 多类模型可能包含 'raptor'/'waterbird' 等子类，全部接受。
        _BIRD_KEYWORDS = {'bird', 'ave', 'raptor', 'waterbird', 'shorebird',
                          'seabird', 'songbird', 'passerine', 'wader', 'waterfowl'}
        bird_cls_ids: set = set()
        try:
            if isinstance(names, dict):
                for k, v in names.items():
                    if any(kw in str(v).lower() for kw in _BIRD_KEYWORDS):
                        bird_cls_ids.add(int(k))
            elif isinstance(names, (list, tuple)):
                for i, v in enumerate(names):
                    if any(kw in str(v).lower() for kw in _BIRD_KEYWORDS):
                        bird_cls_ids.add(int(i))
        except Exception:
            bird_cls_ids = set()
        # 专用单类模型且 names 不含关键词时，默认接受所有类（整个模型就是鸟）
        _bird_accept_all = (not bird_cls_ids) and (names is not None) and (len(names) == 1)
        # COCO 通用模型 fallback
        if not bird_cls_ids and not _bird_accept_all:
            bird_cls_ids = {14}

        try:
            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            mask_polys = getattr(masks, "xy", None) if masks is not None else None

            if boxes is not None and len(boxes) > 0:
                for i, b in enumerate(boxes):
                    if self._cancelled:
                        return

                    try:
                        cls_id = int(b.cls.item())
                        conf = float(b.conf.item())
                        xyxy = b.xyxy[0].cpu().numpy().astype(float)
                        x1, y1, x2, y2 = [int(round(v)) for v in xyxy.tolist()]
                        x1 = max(0, min(W - 1, x1))
                        y1 = max(0, min(H - 1, y1))
                        x2 = max(0, min(W, x2))
                        y2 = max(0, min(H, y2))
                    except Exception:
                        continue

                    if not _bird_accept_all and cls_id not in bird_cls_ids:
                        continue
                    if conf < YOLO_BIRD_MIN_CONF:
                        continue

                    bird_boxes.append((x1, y1, x2, y2, conf))

                    if mask_polys is not None and i < len(mask_polys):
                        poly = _downsample_points_xy(mask_polys[i], max_points=180)
                        if len(poly) >= 3:
                            bird_polys.append(poly)

            bird_boxes = sorted(bird_boxes, key=lambda t: t[4], reverse=True)[:5]
            if bird_boxes:
                print(f'[YOLO] 检测到 {len(bird_boxes)} 只鸟，最高置信度 {bird_boxes[0][4]:.2f}，轮廓多边形 {len(bird_polys)} 个')
            else:
                print(f'[YOLO] 未检测到鸟（阈值 {YOLO_BIRD_MIN_CONF}，图像尺寸 {W}×{H}）')
        except Exception as e:
            print(f'[YOLO] 鸟身结果解析异常：{e}')
            bird_boxes = []
            bird_polys = []

        if self._cancelled:
            return

        try:
            eye_model = _get_model(self._eye_model_path)
            if eye_model is None:
                print(f'[YOLO] 鸟眼模型加载失败：{self._eye_model_path}')
                self.sig_ready.emit(self._path, {"bird_boxes": bird_boxes, "bird_polys": bird_polys, "eye_boxes": []})
                return
        except FileNotFoundError:
            print(f'[YOLO] 鸟眼模型文件不存在：{self._eye_model_path}')
            self.sig_ready.emit(self._path, {"bird_boxes": bird_boxes, "bird_polys": bird_polys, "eye_boxes": []})
            return
        except Exception as e:
            print(f'[YOLO] 鸟眼模型加载异常：{e}')
            self.sig_ready.emit(self._path, {"bird_boxes": bird_boxes, "bird_polys": bird_polys, "eye_boxes": []})
            return

        if self._cancelled:
            return

        try:
            for (x1, y1, x2, y2, _) in bird_boxes:
                if self._cancelled:
                    return

                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(W, x2)
                y2c = min(H, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                crop = img[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                with _YOLO_LOCK:
                    er = eye_model(crop, imgsz=1024, verbose=False)[0]
                eboxes = getattr(er, "boxes", None)
                if eboxes is None or len(eboxes) == 0:
                    continue

                best = None
                best_conf = -1.0
                for eb in eboxes:
                    try:
                        conf = float(eb.conf.item())
                    except Exception:
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best = eb

                if best is None:
                    continue

                if float(best_conf) < YOLO_EYE_MIN_CONF:
                    continue

                xyxy = best.xyxy[0].cpu().numpy().astype(float)
                ex1, ey1, ex2, ey2 = [int(round(v)) for v in xyxy.tolist()]
                ex1 += x1c
                ex2 += x1c
                ey1 += y1c
                ey2 += y1c
                ex1 = max(0, min(W - 1, ex1))
                ey1 = max(0, min(H - 1, ey1))
                ex2 = max(0, min(W, ex2))
                ey2 = max(0, min(H, ey2))
                eye_boxes.append((ex1, ey1, ex2, ey2, float(best_conf)))
            if eye_boxes:
                print(f'[YOLO] 检测到 {len(eye_boxes)} 个鸟眼，最高置信度 {max(e[4] for e in eye_boxes):.2f}')
        except Exception as e:
            print(f'[YOLO] 鸟眼推理异常：{e}')
            eye_boxes = []

        self.sig_ready.emit(self._path, {"bird_boxes": bird_boxes, "bird_polys": bird_polys, "eye_boxes": eye_boxes})


class YoloOverlayManager(QObject):
    sig_ready = Signal(str, object)

    def __init__(self, parent=None, bird_model_path: Optional[str] = None, eye_model_path: Optional[str] = None):
        super().__init__(parent)
        base = Path(__file__).resolve().parent
        self._bird_model = str((base / "models" / "yolov8x-seg.pt").resolve()) if bird_model_path is None else str(bird_model_path)
        self._eye_model = str((base / "models" / "best.pt").resolve()) if eye_model_path is None else str(eye_model_path)

        self._t: Optional[_OverlayThread] = None
        self._running_path: Optional[str] = None
        # 保活列表：旧的已取消线程在此等待自然结束，防止 Python GC 过早销毁
        self._old_threads: List[_OverlayThread] = []

    def request(self, path: str, qimage: QImage):
        p = str(Path(path).resolve())
        self._running_path = p

        # 取消旧线程并移入保活列表（不设 parent，由列表保持引用）
        if self._t is not None:
            try:
                self._t.cancel()
            except Exception:
                pass
            self._old_threads.append(self._t)

        # 清理已结束的旧线程
        self._old_threads = [t for t in self._old_threads if t.isRunning()]

        # 不设 parent=self，由 _old_threads / self._t 持有 Python 引用
        t = _OverlayThread(p, qimage, self._bird_model, self._eye_model)
        t.sig_ready.connect(self._on_ready)
        self._t = t
        t.start()

    def _on_ready(self, path: str, result: dict):
        if self._running_path and str(Path(path).resolve()) != str(Path(self._running_path).resolve()):
            return
        self.sig_ready.emit(path, result or {})

    def shutdown(self):
        # 取消并等待当前线程
        if self._t is not None:
            try:
                self._t.cancel()
                self._t.wait(500)
            except Exception:
                pass
            self._t = None
        self._running_path = None
        # 等待所有旧线程结束，防止 Qt 在销毁时遇到仍在运行的线程
        for t in list(self._old_threads):
            try:
                t.cancel()
                t.wait(300)
            except Exception:
                pass
        self._old_threads.clear()