#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""应用级常量：配色、字体、布局尺寸等。"""

import sys
from qt_compat import QColor

# ── 旗标徽章配色 ─────────────────────────────────────────────────────
CLR_PICK_BG    = QColor( 42, 110,  82, 220)
CLR_PICK_FG    = QColor(210, 235, 220)
CLR_REJECT_BG  = QColor(130,  58,  72, 220)
CLR_REJECT_FG  = QColor(240, 215, 220)
CLR_PICK_STRIP = QColor( 58, 138, 100)
CLR_REJ_STRIP  = QColor(155,  68,  85)

# ── 对焦框配色 ────────────────────────────────────────────────────────
CLR_IN_FOCUS = QColor( 50, 220, 110, 255)
CLR_SELECTED = QColor(158, 131,  74, 210)
CLR_INACTIVE = QColor(100, 110, 130, 150)

# ── 状态栏样式 ────────────────────────────────────────────────────────
SB_FONT_SIZE  = 11
SB_BG         = '#2b2b2b'
SB_BORDER     = '#3a3a3a'
SB_SEP        = '#4a4a4a'
SB_LBL        = '#9a9a9a'
SB_VAL        = '#c8c8c8'
SB_VAL_ACTIVE = '#e0e0e0'
SB_VAL_AF_OK  = '#6db88a'
SB_VAL_LOCK   = '#c9a55a'
SB_VAL_WARN   = '#c9a55a'
SB_VAL_ERR    = '#c47070'
SB_ET_OK      = '#bdbdbd'
SB_ET_NO      = '#8f8f8f'
SB_VAL_PICK   = '#b8b8b8'
SB_VAL_REJ    = '#9a9a9a'

# ── 评分档位配色 ──────────────────────────────────────────────────────
SCORE_CLR_GREEN  = QColor(132, 156, 136)
SCORE_CLR_YELLOW = QColor(166, 154, 118)
SCORE_CLR_RED    = QColor(158, 124, 124)

# ── UI 字体 ───────────────────────────────────────────────────────────
UI_FONT_FAMILY    = 'Microsoft YaHei UI' if sys.platform == 'win32' else 'Arial'
UI_FONT_BASE_SIZE = 10

# ── 扫描批次 / 评分 ───────────────────────────────────────────────────
SCAN_BATCH             = 50
SCORE_VERIFY_TOLERANCE = 0.05
YOLO_BIRD_MIN_CONF     = 0.35

# ── 缩略图条布局 ──────────────────────────────────────────────────────
_STRIP_H      = 90
_CELL_W       = 100
_THUMB_W      = 86
_THUMB_H      = 66
_CELL_PAD_TOP = 4
_LABEL_H      = 14
_LABEL_GAP    = 2

# ── EXIF 叠加条布局 ───────────────────────────────────────────────────
_EXIF_BAR_LINE_H = 24
_EXIF_BAR_PAD_X  = 16
_EXIF_BAR_PAD_Y  = 10
_EXIF_COL_GAP    = 24
