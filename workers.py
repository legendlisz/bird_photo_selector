#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台线程、数据结构和线程生命周期管理。

包含：
  - ImageData        图像数据容器
  - LoadThread       单张 RAW 加载 + EXIF 解析线程
  - ScanThread       目录扫描线程
  - CombinedBatchScoringThread  批量清晰度 + YOLO 融合评分线程
  - PreloadManager   预加载管理器
  - retire_thread / cleanup_zombies_force  线程生命周期助手
"""

import contextlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from qt_compat import QThread, QObject, Signal, _PYSIDE6
from af_parser import (
    AFPoint, AFExtractor, BRAND_BY_EXT,
    _first_int, _first_str, _fmt_focus_dist,
)
from utils import (
    natural_key, ScoreCache,
    _qimage_to_gray, _qimage_to_bgr,
    calculate_image_sharpness, compute_center_sharpness_qimage,
)
from raw_reader import RAWReader
from af_parser import SUPPORTED_EXTS


# ════════════════════════════════════════════════════════════════════
#  § 数据结构
# ════════════════════════════════════════════════════════════════════

@dataclass
class ImageData:
    qimage: Any; disp_w: int; disp_h: int; ref_w: int; ref_h: int
    af_points: List[AFPoint] = field(default_factory=list)
    af_msg: str = ""; exif: Dict[str, Any] = field(default_factory=dict)
    sharpness: float = 0.0; pixmap: Any = None


# ════════════════════════════════════════════════════════════════════
#  § 线程生命周期助手
# ════════════════════════════════════════════════════════════════════

_ZOMBIES: List[QThread] = []


def retire_thread(t: Optional[QThread]):
    """将线程'退休'到全局僵尸池，防止其 Python 对象被过早销毁。"""
    if not t: return
    if t.isRunning():
        if hasattr(t, 'cancel'): t.cancel()
    import warnings as _warnings
    for _sig in ('sig_done','sig_error','sig_progress','sig_ready','sig_loaded','sig_batch'):
        sig_obj = getattr(t, _sig, None)
        if sig_obj is None: continue
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            try: sig_obj.disconnect()
            except Exception: pass
    global _ZOMBIES
    if t not in _ZOMBIES: _ZOMBIES.append(t)


def cleanup_zombies_force():
    global _ZOMBIES
    _ZOMBIES = [t for t in _ZOMBIES if t.isRunning()]


# ════════════════════════════════════════════════════════════════════
#  § 目录扫描线程
# ════════════════════════════════════════════════════════════════════

_SCAN_BATCH = 50

def is_scan_noise_file(name: str) -> bool:
    n = str(name)
    return n.startswith('._') or n.startswith('.')


class ScanThread(QThread):
    sig_batch = Signal(list, int)
    sig_done  = Signal(list)

    def __init__(self, folder: str):
        super().__init__(); self._folder = folder; self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        found: List[str] = []; batch: List[str] = []
        try:
            for entry in os.scandir(self._folder):
                if self._cancelled: return
                name = entry.name
                if is_scan_noise_file(name): continue
                if entry.is_file() and Path(name).suffix.lower() in SUPPORTED_EXTS:
                    p = str(Path(entry.path).resolve())
                    found.append(p); batch.append(p)
                    if len(batch) >= _SCAN_BATCH:
                        self.sig_batch.emit(batch, len(found)); batch = []
        except PermissionError:
            pass
        if self._cancelled: return
        if batch: self.sig_batch.emit(batch, len(found))
        found.sort(key=lambda p: natural_key(os.path.basename(p)))
        self.sig_done.emit(found)


# ════════════════════════════════════════════════════════════════════
#  § 单张图像加载线程
# ════════════════════════════════════════════════════════════════════

class LoadThread(QThread):
    sig_done  = Signal(object)
    sig_error = Signal(str)

    def __init__(self, path: str):
        super().__init__(); self._path = path; self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        if self._cancelled: return
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_qi   = ex.submit(RAWReader.read_qimage, self._path)
                f_exif = ex.submit(AFExtractor.extract,   self._path)
                qi, err              = f_qi.result()
                af_pts, af_msg, exif = f_exif.result()
            if self._cancelled: return
            if qi is None or qi.isNull():
                self.sig_error.emit(err or '读取失败'); return

            fm_raw = _first_str(exif,
                'MakerNotes:FocusMode','Nikon:FocusMode','Canon:FocusMode',
                'Sony:FocusMode','Olympus:FocusMode',
                'ExifIFD:FocusMode','EXIF:FocusMode','FocusMode')
            if fm_raw:
                if re.fullmatch(r'\d+(?:\s+\d+)*', str(fm_raw).strip()):
                    fm_txt = AFExtractor._run_focus_mode_text(self._path)
                    exif['_focus_mode'] = fm_txt if fm_txt else str(fm_raw).strip()
                else:
                    exif['_focus_mode'] = str(fm_raw).strip()

            dist_raw = _first_str(exif,
                'Sony:Focus Distance 2','FocusDistance2','Composite:FocusDistance2',
                'Nikon:FocusDistance','FocusDistance','Composite:FocusDistance',
                'Canon:FocusDistance','Olympus:FocusDistance',
                'MakerNotes:FocusDistance','Composite:HyperfocalDistance')
            fd_fmt = _fmt_focus_dist(dist_raw)
            if fd_fmt: exif['_focus_dist'] = fd_fmt
            if af_pts: exif['_af_point_count'] = len(af_pts)

            brand = BRAND_BY_EXT.get(Path(self._path).suffix.lower(), '')
            if brand == 'OM':
                ref_w = exif.get('_OM_ref_w', 1); ref_h = exif.get('_OM_ref_h', 1)
            else:
                ref_w = (_first_int(exif,
                    'Nikon:AFImageWidth','MakerNotes:AFImageWidth',
                    'EXIF:ExifImageWidth','ExifIFD:ExifImageWidth','File:ImageWidth') or qi.width())
                ref_h = (_first_int(exif,
                    'Nikon:AFImageHeight','MakerNotes:AFImageHeight',
                    'EXIF:ExifImageHeight','ExifIFD:ExifImageHeight','File:ImageHeight') or qi.height())

            gray = _qimage_to_gray(qi)
            if gray is not None:
                sharpness = calculate_image_sharpness(gray, af_pts, ref_w, ref_h, qi.width(), qi.height())
            else:
                sharpness = compute_center_sharpness_qimage(qi)
            if self._cancelled: return
            self.sig_done.emit(ImageData(
                qimage=qi, disp_w=qi.width(), disp_h=qi.height(),
                ref_w=ref_w, ref_h=ref_h, af_points=af_pts,
                af_msg=af_msg, exif=exif, sharpness=sharpness, pixmap=None))
        except Exception as e:
            if not self._cancelled: self.sig_error.emit(f'加载线程异常: {e}')


# ════════════════════════════════════════════════════════════════════
#  § YOLO 推理辅助（供批量评分线程调用）
# ════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _null_ctx():
    yield


def _run_yolo_on_image(img: np.ndarray, bird_model, eye_model) -> dict:
    """在 BGR numpy 图像上运行鸟身分割 + 鸟眼检测，返回与 yolo_overlay 同结构的 dict。"""
    try:
        from yolo_overlay import _downsample_points_xy, _YOLO_LOCK as _lock, YOLO_BIRD_MIN_CONF, YOLO_EYE_MIN_CONF
    except ImportError:
        _lock = None; YOLO_BIRD_MIN_CONF = 0.35; YOLO_EYE_MIN_CONF = 0.30
        def _downsample_points_xy(xy, max_points=160): return []

    H, W = img.shape[:2]
    bird_boxes: List[Tuple[int,int,int,int,float]] = []
    bird_polys: List[List[Tuple[int,int]]]         = []
    eye_boxes:  List[Tuple[int,int,int,int,float]] = []

    lock_ctx = _lock if _lock else _null_ctx()

    with lock_ctx:
        res = bird_model(img, imgsz=1024, verbose=False)[0]

    names = getattr(bird_model, 'names', None)
    bird_cls_id = None
    try:
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == 'bird': bird_cls_id = int(k); break
        elif isinstance(names, (list, tuple)):
            for i2, v in enumerate(names):
                if str(v).lower() == 'bird': bird_cls_id = int(i2); break
    except Exception:
        bird_cls_id = None

    boxes = getattr(res, 'boxes', None)
    masks = getattr(res, 'masks', None)
    mask_polys = getattr(masks, 'xy', None) if masks is not None else None

    if boxes is not None and len(boxes) > 0:
        for bi, b in enumerate(boxes):
            try:
                cls_id = int(b.cls.item()); conf = float(b.conf.item())
                xyxy = b.xyxy[0].cpu().numpy().astype(float)
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy.tolist()]
                x1=max(0,min(W-1,x1)); y1=max(0,min(H-1,y1))
                x2=max(0,min(W,x2));   y2=max(0,min(H,y2))
            except Exception: continue
            if (bird_cls_id is not None and cls_id != bird_cls_id) or \
               (bird_cls_id is None     and cls_id != 14): continue
            if conf < YOLO_BIRD_MIN_CONF: continue
            bird_boxes.append((x1,y1,x2,y2,conf))
            if mask_polys is not None and bi < len(mask_polys):
                poly = _downsample_points_xy(mask_polys[bi], max_points=180)
                if len(poly) >= 3: bird_polys.append(poly)

    bird_boxes = sorted(bird_boxes, key=lambda t: t[4], reverse=True)[:5]

    for (x1,y1,x2,y2,_) in bird_boxes:
        x1c=max(0,x1); y1c=max(0,y1); x2c=min(W,x2); y2c=min(H,y2)
        if x2c<=x1c or y2c<=y1c: continue
        crop=img[y1c:y2c,x1c:x2c]
        if crop.size==0: continue
        with lock_ctx:
            er=eye_model(crop,imgsz=1024,verbose=False)[0]
        eboxes=getattr(er,'boxes',None)
        if eboxes is None or len(eboxes)==0: continue
        best=None; best_conf=-1.0
        for eb in eboxes:
            try: c=float(eb.conf.item())
            except Exception: continue
            if c>best_conf: best_conf=c; best=eb
        if best is None or float(best_conf) < YOLO_EYE_MIN_CONF: continue
        xyxy=best.xyxy[0].cpu().numpy().astype(float)
        ex1,ey1,ex2,ey2=[int(round(v)) for v in xyxy.tolist()]
        ex1+=x1c; ex2+=x1c; ey1+=y1c; ey2+=y1c
        ex1=max(0,min(W-1,ex1)); ey1=max(0,min(H-1,ey1))
        ex2=max(0,min(W,ex2));   ey2=max(0,min(H,ey2))
        eye_boxes.append((ex1,ey1,ex2,ey2,float(best_conf)))

    return {'bird_boxes':bird_boxes,'bird_polys':bird_polys,'eye_boxes':eye_boxes}


# ════════════════════════════════════════════════════════════════════
#  § 批量清晰度 + YOLO 融合评分线程
# ════════════════════════════════════════════════════════════════════

class CombinedBatchScoringThread(QThread):
    """单次 RAW 读取同时完成清晰度评分 + YOLO 融合评分。

    每张图只解码一次 RAW、只调用一次 exiftool。
    YOLO 模型不可用时自动降级为纯清晰度模式。
    已缓存的字段自动跳过，不重复计算。
    """
    sig_progress = Signal(str, float, float, int, int)
    sig_done     = Signal(int)

    def __init__(self, paths: List[str], cache: ScoreCache,
                 force_sharp: bool = False, force_fused: bool = False):
        super().__init__()
        self._paths       = list(paths)
        self._cache       = cache
        self._force_sharp = force_sharp
        self._force_fused = force_fused
        self._cancelled   = False

        base = Path(os.path.abspath(__file__)).parent
        self._bird_model = str((base / 'models' / 'yolov8x-seg.pt').resolve())
        self._eye_model  = str((base / 'models' / 'best.pt').resolve())

    def cancel(self): self._cancelled = True

    def run(self):
        need_sharp = set(); need_fused = set()
        for p in self._paths:
            if self._force_sharp or not self._cache.has(p):        need_sharp.add(p)
            if self._force_fused or not self._cache.has_fused(p):  need_fused.add(p)
        to_score = [p for p in self._paths if p in need_sharp or p in need_fused]
        if not to_score:
            self.sig_done.emit(0); return

        bird_model = eye_model = None; yolo_ok = False
        try:
            from score_fusion import compute_fused_score as _cfs
        except Exception:
            _cfs = None
        if need_fused and _cfs is not None:
            try:
                from ultralytics import YOLO as _YOLO
                try:
                    from yolo_overlay import _get_model as _yolo_get_model
                except ImportError:
                    _yolo_get_model = None
                if os.path.isfile(self._bird_model) and os.path.isfile(self._eye_model):
                    if _yolo_get_model is not None:
                        bird_model = _yolo_get_model(self._bird_model)
                        eye_model  = _yolo_get_model(self._eye_model)
                    else:
                        bird_model = _YOLO(self._bird_model)
                        eye_model  = _YOLO(self._eye_model)
                    yolo_ok = bird_model is not None and eye_model is not None
            except Exception as e:
                print(f'[CombinedScoring] YOLO 不可用，跳过融合评分: {e}')

        total = len(to_score); done = 0; save_every = 5; CHUNK = 50

        def _calc_sharp(full_qi, af_pts, ref_w, ref_h):
            full_bgr = _qimage_to_bgr(full_qi)
            if full_bgr is None or full_bgr.size == 0:
                return compute_center_sharpness_qimage(full_qi), None
            gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
            if af_pts:
                s = calculate_image_sharpness(gray, af_pts, ref_w, ref_h, full_qi.width(), full_qi.height())
            else:
                s = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            return s, full_bgr

        def _calc_yolo(thumb_qi, bm, em):
            img = _qimage_to_bgr(thumb_qi)
            if img is None or img.size == 0:
                return {'bird_boxes':[],'bird_polys':[],'eye_boxes':[]}
            return _run_yolo_on_image(img, bm, em)

        def _decode_one(path, exif_map):
            thumb_qi, full_qi, err = RAWReader.read_qimage_thumb_and_full(path)
            exif = exif_map.get(str(Path(path).resolve()), {}) or {}
            if not exif:
                exif = AFExtractor._run(path)
            return path, thumb_qi, full_qi, exif, err

        try:
            for base_i in range(0, total, CHUNK):
                if self._cancelled: break
                chunk_paths = to_score[base_i:base_i + CHUNK]
                exif_map: Dict[str, dict] = {}
                # Use EXIFTOOL_PATH at runtime (set by af_parser.set_exiftool_path)
                import af_parser as _af_mod
                if _af_mod.EXIFTOOL_PATH and chunk_paths:
                    exif_map = AFExtractor.extract_batch(chunk_paths)

                io_pool = ThreadPoolExecutor(max_workers=2)
                ordered_futures = [io_pool.submit(_decode_one, p, exif_map) for p in chunk_paths]

                for j, fut in enumerate(ordered_futures):
                    if self._cancelled: break
                    path = chunk_paths[j]
                    sharpness = 0.0; fused_score = 0.0
                    do_sharp = path in need_sharp; do_fused = path in need_fused and yolo_ok

                    try:
                        _, thumb_qi, full_qi, exif, err = fut.result()
                        if full_qi is None or full_qi.isNull(): raise RuntimeError(err or '读取失败')

                        af_pts, _ = AFExtractor._parse(path, exif) if exif else ([], '')
                        brand = BRAND_BY_EXT.get(Path(path).suffix.lower(), '')
                        if brand == 'OM':
                            ref_w = exif.get('_OM_ref_w', 1); ref_h = exif.get('_OM_ref_h', 1)
                        else:
                            ref_w = (_first_int(exif,
                                'Nikon:AFImageWidth','MakerNotes:AFImageWidth',
                                'EXIF:ExifImageWidth','ExifIFD:ExifImageWidth','File:ImageWidth')
                                or full_qi.width())
                            ref_h = (_first_int(exif,
                                'Nikon:AFImageHeight','MakerNotes:AFImageHeight',
                                'EXIF:ExifImageHeight','ExifIFD:ExifImageHeight','File:ImageHeight')
                                or full_qi.height())

                        full_bgr_cache = None; yolo = None
                        if do_sharp or do_fused:
                            cached_sharp = self._cache.get(path)
                            if (not do_sharp) and cached_sharp is not None and float(cached_sharp) > 0:
                                sharpness = float(cached_sharp)
                            else:
                                if do_fused:
                                    with ThreadPoolExecutor(max_workers=2) as compute_pool:
                                        fut_sharp = compute_pool.submit(_calc_sharp, full_qi, af_pts, ref_w, ref_h)
                                        fut_yolo  = compute_pool.submit(_calc_yolo, thumb_qi, bird_model, eye_model)
                                        sharpness, full_bgr_cache = fut_sharp.result()
                                        yolo = fut_yolo.result()
                                else:
                                    sharpness, full_bgr_cache = _calc_sharp(full_qi, af_pts, ref_w, ref_h)

                        if do_sharp: self._cache.set(path, sharpness)

                        if do_fused:
                            if yolo is None:
                                img = full_bgr_cache if full_bgr_cache is not None else _qimage_to_bgr(thumb_qi)
                                if img is None or img.size == 0:
                                    yolo = {'bird_boxes':[],'bird_polys':[],'eye_boxes':[]}
                                else:
                                    yolo = _run_yolo_on_image(img, bird_model, eye_model)
                            fused_score, _ = _cfs(
                                full_qi, af_pts, ref_w, ref_h, float(sharpness or 0.0), yolo)
                            bird_box_str = ""
                            if yolo.get('bird_boxes'):
                                b = yolo['bird_boxes'][0]
                                bird_box_str = f"{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}"
                            self._cache.set_fused(path, fused_score, bird_box=bird_box_str)

                    except Exception as e:
                        print(f'[CombinedScoring] {os.path.basename(path)}: {e}')

                    if self._cancelled: break
                    done += 1
                    if done % save_every == 0: self._cache.save()
                    self.sig_progress.emit(
                        os.path.basename(path), float(sharpness), float(fused_score),
                        base_i + j + 1, total)

                io_pool.shutdown(wait=False)

        except Exception as e:
            print(f'[CombinedScoring] 线程异常: {e}')

        self._cache.save_if_dirty()
        self.sig_done.emit(done)


# ════════════════════════════════════════════════════════════════════
#  § 预加载管理器
# ════════════════════════════════════════════════════════════════════

class PreloadManager(QObject):
    sig_loaded = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache: Dict[str, ImageData]       = {}
        self._access_times: Dict[str, float]    = {}
        self._max_cache: int                    = 10
        self._threads: Dict[str, LoadThread]    = {}
        self._queue: List[str]                  = []

    def get(self, path: str) -> Optional[ImageData]:
        if path in self._cache:
            self._access_times[path] = datetime.now().timestamp()
            return self._cache.get(path)
        return None

    def clear(self):
        for t in self._threads.values(): retire_thread(t)
        self._threads.clear(); self._cache.clear()
        self._access_times.clear(); self._queue.clear()

    def remove(self, path: str):
        if path in self._cache:        del self._cache[path]
        if path in self._access_times: del self._access_times[path]
        if path in self._threads:
            t = self._threads.pop(path); retire_thread(t)

    def preload(self, paths: List[str]):
        new_queue = [p for p in paths if p not in self._cache]
        to_cancel = [p for p in self._threads if p not in new_queue]
        for p in to_cancel: t = self._threads.pop(p); retire_thread(t)
        self._queue = new_queue; self._start_next()

    def _start_next(self):
        if not self._queue or len(self._threads) >= 3: return
        path = self._queue.pop(0)
        if path in self._threads: self._start_next(); return
        t = LoadThread(path)
        t.sig_done .connect(lambda data, p=path: self._on_done(p, data))
        t.sig_error.connect(lambda err,  p=path: self._on_error(p, err))
        self._threads[path] = t; t.start()

    def _on_done(self, path, data):
        if path in self._threads:
            t = self._threads.pop(path); retire_thread(t)
        self._cache[path] = data
        self._access_times[path] = datetime.now().timestamp()
        if len(self._cache) > self._max_cache:
            sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
            for i in range(len(self._cache) - self._max_cache):
                k = sorted_keys[i]; del self._cache[k]; del self._access_times[k]
        self.sig_loaded.emit(path, data); self._start_next()

    def _on_error(self, path, err):
        if path in self._threads: t = self._threads.pop(path); retire_thread(t)
        self._start_next()
