#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXIF 字段格式化工具：快门、光圈、焦距等人类可读字符串转换。"""

import re
from typing import List

from af_parser import _first_str

# ── 格式化辅助 ────────────────────────────────────────────────────────

def _fmt_shutter(raw: str) -> str:
    try:
        v = float(raw)
        if v <= 0: return raw
        if v >= 1: return f'{v:.1f}s'
        recip = round(1.0 / v)
        return f'1/{recip}s'
    except Exception:
        return raw


def _fmt_aperture(raw: str) -> str:
    try:
        return f'f/{float(raw):.1f}'
    except Exception:
        return raw


def _fmt_focal(raw: str) -> str:
    try:
        v = float(raw)
        return f'{v:.0f}mm'
    except Exception:
        return raw


_EXP_PROG_MAP = {
    '0': '未定义', '1': '手动',    '2': '程序自动',
    '3': '光圈优先', '4': '快门优先',
    '5': '创意',    '6': '动作',   '7': '人像', '8': '风景',
}

# ── EXIF 叠加行构建 ────────────────────────────────────────────────────

def get_exif_overlay_rows(exif: dict) -> List[List[tuple]]:
    """从 exif dict 构建 [(label, value), ...] 的行列表，供 ImageView 绘制。"""
    if not exif:
        return []
    row1, row2 = [], []

    cam = _first_str(exif, 'IFD0:Model', 'EXIF:Model', 'Model')
    if cam:
        cam = re.sub(
            r'^(?:NIKON\s+|Canon\s+|SONY\s+|OM SYSTEM\s+|OLYMPUS\s+)', '',
            cam, flags=re.I).strip()
        row1.append(('相机', cam))

    lens = _first_str(exif,
        'ExifIFD:LensModel', 'EXIF:LensModel', 'Composite:LensID', 'MakerNotes:LensID',
        'Nikon:Lens', 'Canon:LensModel', 'Sony:LensID', 'Olympus:LensModel', 'LensModel')
    if lens:
        row1.append(('镜头', lens))

    fd = exif.get('_focus_dist')
    if fd:
        row1.append(('对焦距离', fd))

    iso = _first_str(exif,
        'ExifIFD:ISO', 'EXIF:ISO', 'MakerNotes:ISO', 'Nikon:ISO',
        'Sony:ISO', 'Canon:ISO', 'Olympus:ISOValue', 'ISO')
    if iso:
        row2.append(('ISO', iso))

    et = _first_str(exif,
        'ExifIFD:ExposureTime', 'EXIF:ExposureTime',
        'ExposureTime', 'Composite:ShutterSpeed')
    if et:
        row2.append(('快门', _fmt_shutter(et)))

    fn = _first_str(exif,
        'ExifIFD:FNumber', 'EXIF:FNumber',
        'Composite:Aperture', 'MakerNotes:Aperture')
    if fn:
        row2.append(('光圈', _fmt_aperture(fn)))

    fl = _first_str(exif,
        'ExifIFD:FocalLength', 'EXIF:FocalLength',
        'Composite:FocalLength35efl', 'FocalLength')
    if fl:
        row2.append(('焦距', _fmt_focal(fl.split()[0])))

    ep = _first_str(exif,
        'ExifIFD:ExposureProgram', 'EXIF:ExposureProgram', 'ExposureProgram')
    if ep:
        row2.append(('曝光', _EXP_PROG_MAP.get(ep.strip(), ep)))

    fm = exif.get('_focus_mode') or _first_str(exif,
        'MakerNotes:FocusMode', 'Nikon:FocusMode', 'Canon:FocusMode',
        'Sony:FocusMode', 'Olympus:FocusMode',
        'ExifIFD:FocusMode', 'EXIF:FocusMode', 'FocusMode')
    if fm:
        row2.append(('对焦模式', fm))

    dt = _first_str(exif,
        'ExifIFD:DateTimeOriginal', 'EXIF:DateTimeOriginal',
        'DateTimeOriginal', 'IFD0:DateTime')
    if dt:
        dt_fmt = re.sub(r'^(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', dt[:19])
        row2.append(('时间', dt_fmt))

    rows = []
    if row1:
        rows.append(row1)
    if row2:
        rows.append(row2)
    return rows
