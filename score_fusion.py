#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _load_qt_qimage():
    try:
        from PySide6.QtGui import QImage
        return QImage
    except Exception:
        from PyQt5.QtGui import QImage
        return QImage


QImage = _load_qt_qimage()

from utils import _qimage_to_gray


def _lap_var(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    try:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _lap_var_mask(gray: np.ndarray, mask: np.ndarray) -> float:
    if gray is None or gray.size == 0 or mask is None or mask.size == 0:
        return 0.0
    try:
        m = mask.astype(np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape[:2] != gray.shape[:2]:
            return 0.0
        sel = m > 0
        if int(sel.sum()) < 800:
            return 0.0
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        vals = lap[sel]
        return float(vals.var()) if vals.size > 0 else 0.0
    except Exception:
        return 0.0


def _poly_mask(h: int, w: int, polys: List[List[Tuple[int, int]]]) -> Optional[np.ndarray]:
    if not polys or h <= 0 or w <= 0:
        return None
    try:
        best = None
        best_area = -1.0
        for poly in polys:
            if not poly or len(poly) < 3:
                continue
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            area = float(cv2.contourArea(pts))
            if area > best_area:
                best_area = area
                best = pts
        if best is None or best_area <= 0:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [best], 1)
        return mask
    except Exception:
        return None


def _expand_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int, factor: float) -> Tuple[int, int, int, int]:
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    s = math.sqrt(max(1e-6, factor))
    nbw = bw * s
    nbh = bh * s
    nx1 = int(max(0, round(cx - nbw * 0.5)))
    ny1 = int(max(0, round(cy - nbh * 0.5)))
    nx2 = int(min(w, round(cx + nbw * 0.5)))
    ny2 = int(min(h, round(cy + nbh * 0.5)))
    return nx1, ny1, nx2, ny2


def _norm01(v: float, vref: float = 450.0) -> float:
    v = float(v or 0.0)
    if v <= 0:
        return 0.0
    try:
        return float(min(1.0, max(0.0, math.log1p(v) / math.log1p(vref))))
    except Exception:
        return 0.0


def _af_boxes_from_points(
    af_points: List[Any],
    ref_w: int,
    ref_h: int,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    if not af_points or img_w <= 0 or img_h <= 0:
        return []

    rw = int(ref_w or 0) or img_w
    rh = int(ref_h or 0) or img_h
    if rw <= 0 or rh <= 0:
        return []

    sx = img_w / float(rw)
    sy = img_h / float(rh)

    boxes: List[Tuple[int, int, int, int]] = []
    for pt in af_points:
        try:
            cx = float(getattr(pt, 'cx')) * sx
            cy = float(getattr(pt, 'cy')) * sy
            bw = float(getattr(pt, 'w')) * sx
            bh = float(getattr(pt, 'h')) * sy
        except Exception:
            continue

        x1 = int(round(cx - bw / 2.0))
        y1 = int(round(cy - bh / 2.0))
        x2 = int(round(cx + bw / 2.0))
        y2 = int(round(cy + bh / 2.0))

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def _overlap_ratio_eye_in_af(
    eye_box: Optional[Tuple[int, int, int, int]],
    af_boxes: List[Tuple[int, int, int, int]],
) -> float:
    if not eye_box or not af_boxes:
        return 0.0

    ex1, ey1, ex2, ey2 = eye_box
    eye_area = float(max(0, ex2 - ex1) * max(0, ey2 - ey1))
    if eye_area <= 0:
        return 0.0

    best = 0.0
    for (ax1, ay1, ax2, ay2) in af_boxes:
        ix1 = max(ex1, ax1)
        iy1 = max(ey1, ay1)
        ix2 = min(ex2, ax2)
        iy2 = min(ey2, ay2)
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            continue
        inter = float(iw * ih)
        best = max(best, inter / eye_area)
    return float(min(1.0, max(0.0, best)))


def compute_fused_score(
    qimage: QImage,
    af_points: List[Any],
    ref_w: int,
    ref_h: int,
    af_sharpness: float,
    yolo_result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    gray = _qimage_to_gray(qimage)
    if gray is None:
        return 0.0, {"reason": "no_gray"}

    h, w = gray.shape[:2]
    img_area = float(w * h) if w > 0 and h > 0 else 1.0

    y = yolo_result or {}
    bird_polys = y.get("bird_polys") or []
    bird_boxes = y.get("bird_boxes") or []
    eye_boxes = y.get("eye_boxes") or []

    v_af = float(af_sharpness or 0.0)
    s_af = _norm01(v_af)

    mask = _poly_mask(h, w, bird_polys) if bird_polys else None
    bird_area_ratio = float(mask.sum()) / img_area if mask is not None else 0.0
    v_bird = _lap_var_mask(gray, mask) if mask is not None else 0.0
    s_bird = _norm01(v_bird)

    v_eye = 0.0
    s_eye = 0.0
    eye_conf = 0.0
    eye_area = 0.0
    if eye_boxes:
        try:
            bx = max(eye_boxes, key=lambda t: float(t[4] if len(t) > 4 else 0.0))
            ex1, ey1, ex2, ey2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            eye_conf = float(bx[4] if len(bx) > 4 else 0.0)
            ex1, ey1, ex2, ey2 = _expand_box(ex1, ey1, ex2, ey2, w, h, factor=1.8)
            if ex2 > ex1 and ey2 > ey1:
                eye_roi = gray[ey1:ey2, ex1:ex2]
                eye_area = float((ex2 - ex1) * (ey2 - ey1))
                v_eye = _lap_var(eye_roi)
                s_eye = _norm01(v_eye)
        except Exception:
            pass

    if af_points:
        try:
            r_af = 1.0 if any(bool(getattr(p, 'in_focus', False)) for p in af_points) else 0.7
        except Exception:
            r_af = 0.7
    else:
        r_af = 0.3

    bird_conf = 0.0
    if bird_boxes:
        try:
            bird_conf = float(max(bird_boxes, key=lambda t: float(t[4] if len(t) > 4 else 0.0))[4])
        except Exception:
            bird_conf = 0.0
    if bird_conf <= 0.0 and mask is not None:
        bird_conf = 0.6

    size_factor = 0.0
    if bird_area_ratio > 0:
        size_factor = float(min(1.0, max(0.0, math.sqrt(bird_area_ratio / 0.08))))
    r_bird = float(max(0.0, min(1.0, bird_conf * size_factor)))

    if bird_area_ratio > 0 and eye_area > 0:
        denom_area = max(1.0, bird_area_ratio * img_area)
        eye_ratio_in_bird = eye_area / denom_area
        eye_ratio_in_bird = min(1.0, max(0.0, float(eye_ratio_in_bird)))
        eye_size_factor = math.sqrt(max(0.0, eye_ratio_in_bird / 0.02))
        eye_size_factor = float(min(1.0, max(0.0, eye_size_factor)))
        r_eye = float(eye_conf) * float(eye_size_factor)
    else:
        r_eye = float(eye_conf) * 0.3
    r_eye = float(min(1.0, max(0.0, r_eye)))

    eye_box_overlap = None
    if eye_boxes:
        try:
            bx = max(eye_boxes, key=lambda t: float(t[4] if len(t) > 4 else 0.0))
            eye_box_overlap = (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))
        except Exception:
            eye_box_overlap = None

    af_boxes = _af_boxes_from_points(af_points, ref_w, ref_h, w, h)
    s_overlap = _overlap_ratio_eye_in_af(eye_box_overlap, af_boxes)

    term_eye = s_eye * r_eye
    term_overlap = s_overlap * r_eye * r_af
    term_af = s_af * r_af
    term_bird = s_bird * r_bird

    W_EYE = 0.45
    W_OVERLAP = 0.20
    W_AF = 0.25
    W_BIRD = 0.10

    # 直接计算总分，不使用质量分
    total = (
        W_EYE * term_eye
        + W_OVERLAP * term_overlap
        + W_AF * term_af
        + W_BIRD * term_bird
    )

    if 0 < bird_area_ratio < 0.01:
        total *= 0.85

    total = float(max(0.0, min(1.0, total)))
    score = total * 1000.0

    details = {
        'score': score,
        's_eye': s_eye,
        's_overlap': s_overlap,
        's_af': s_af,
        's_bird': s_bird,
        'v_eye': v_eye,
        'v_af': v_af,
        'v_bird': v_bird,
        'r_eye': r_eye,
        'r_af': r_af,
        'r_bird': r_bird,
        'bird_area_ratio': bird_area_ratio,
        'eye_conf': eye_conf,
        'bird_conf': bird_conf,
        'W_EYE': W_EYE,
        'W_OVERLAP': W_OVERLAP,
        'W_AF': W_AF,
        'W_BIRD': W_BIRD,
    }
    return score, details