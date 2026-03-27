#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ImageView：主图像显示控件，含 AF / YOLO / EXIF 叠加层绘制。"""

from typing import Any, Dict, List, Optional

from qt_compat import (
    QWidget, QMenu, QFont, QColor, QPainter, QPen, QBrush,
    QPoint, QRectF, QFontMetrics, QLinearGradient, QCursor,
    Signal, Qt,
    _PYSIDE6, _STRONG_FOCUS, _SP_EXPAND, _CTX_CUSTOM,
    _ANTIALIASING, _ALIGN_CENTER, _SOLID_LINE, _NO_BRUSH,
    _KEEP_AR, _SMOOTH_XFORM,
    _LMB, _ARROW_CUR, _HAND_CUR, _ROUND_CAP,
    _K_LEFT, _K_RIGHT, _K_ESC, _K_DEL, _K_BACK,
)
from app_constants import (
    SB_BG,
    CLR_PICK_BG, CLR_PICK_FG, CLR_REJECT_BG,
    CLR_IN_FOCUS, CLR_SELECTED, CLR_INACTIVE,
    SCORE_CLR_GREEN, SCORE_CLR_YELLOW, SCORE_CLR_RED,
    UI_FONT_FAMILY,
    _EXIF_BAR_LINE_H, _EXIF_BAR_PAD_X, _EXIF_BAR_PAD_Y, _EXIF_COL_GAP,
)
from exif_utils import get_exif_overlay_rows
from utils import FLAG_NONE, FLAG_PICK, FLAG_REJECT, SHARP_HI, SHARP_LO


class ImageView(QWidget):
    sig_prev              = Signal()
    sig_next              = Signal()
    sig_wheel_steps       = Signal(int)
    sig_zoom_chg          = Signal()
    sig_delete_current    = Signal()
    sig_exif_toggled      = Signal(bool)
    sig_af_toggled        = Signal(bool)
    sig_yolo_bird_toggled = Signal(bool)
    sig_yolo_eye_toggled  = Signal(bool)
    sig_flag_pick         = Signal()
    sig_flag_reject       = Signal()
    sig_reveal_current    = Signal()
    sig_rescore_selected  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(_STRONG_FOCUS)
        self.setStyleSheet(f"background:{SB_BG};")
        self.setSizePolicy(_SP_EXPAND, _SP_EXPAND)
        self.setContextMenuPolicy(_CTX_CUSTOM)
        self.customContextMenuRequested.connect(self._ctx_menu)

        self._pixmap             = None
        self._mode               = 'fit'
        self._offset             = QPoint(0, 0)
        self._press_pos          = None
        self._drag_on            = False
        self._loading            = False
        self._af_points: list    = []
        self._af_ref_w: int      = 0
        self._af_ref_h: int      = 0
        self._sharpness: float   = 0.0
        self._fused_score: float = 0.0
        self._ctx_fname_str: str = ''
        self._show_exif: bool    = False
        self._show_af: bool      = True
        self._show_yolo_bird: bool = True
        self._show_yolo_eye: bool  = True
        self._exif_data: Dict[str, Any] = {}
        self._flag: int          = FLAG_NONE
        self._cached_fit_pm      = None

        self._yolo_bird_boxes: List[tuple] = []
        self._yolo_bird_polys: List[list]  = []
        self._yolo_eye_boxes:  List[tuple] = []

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def zoom_mode(self):
        return self._mode

    # ── 公开设置方法 ──────────────────────────────────────────────────

    def set_flag(self, flag: int):
        self._flag = flag; self.update()

    def set_af_info(self, pts, rw, rh):
        self._af_points = pts; self._af_ref_w = rw; self._af_ref_h = rh

    def set_sharpness(self, val: float, source: str = 'live'):
        self._sharpness = max(0.0, val); self.update()

    def set_fused_score(self, val: float):
        self._fused_score = max(0.0, float(val or 0.0)); self.update()

    def set_exif_data(self, exif: Dict[str, Any]):
        self._exif_data = exif or {}; self.update()

    def set_show_exif(self, v: bool):
        self._show_exif = v; self.update()

    def set_show_af(self, v: bool):
        self._show_af = v; self.update()

    def set_show_yolo_bird(self, v: bool):
        self._show_yolo_bird = v; self.update()

    def set_show_yolo_eye(self, v: bool):
        self._show_yolo_eye = v; self.update()

    def load_pixmap(self, pm):
        self._pixmap = pm; self._mode = 'fit'; self._offset = QPoint(0, 0)
        self._loading = False; self._af_points = []; self._af_ref_w = 0; self._af_ref_h = 0
        self._sharpness = 0.0; self._fused_score = 0.0; self._flag = FLAG_NONE
        self._cached_fit_pm = None
        self._yolo_bird_boxes = []; self._yolo_bird_polys = []; self._yolo_eye_boxes = []
        self.update()

    def set_loading(self, v):
        self._loading = v
        if v:
            self._sharpness = 0.0; self._fused_score = 0.0; self._exif_data = {}
        self.update()

    def enter_100_at_af(self):
        if not self._pixmap: return
        af = self._primary_af()
        ix = af[0] if af else self._pixmap.width()  / 2
        iy = af[1] if af else self._pixmap.height() / 2
        self._offset = QPoint(int(self.width() / 2 - ix), int(self.height() / 2 - iy))
        self._clamp(); self._mode = '100'; self.update(); self.sig_zoom_chg.emit()

    def clear_yolo_overlays(self):
        self._yolo_bird_boxes = []; self._yolo_bird_polys = []; self._yolo_eye_boxes = []
        self.update()

    def set_yolo_overlays(self, bird_boxes, bird_polys, eye_boxes):
        self._yolo_bird_boxes = list(bird_boxes or [])
        self._yolo_bird_polys = list(bird_polys or [])
        self._yolo_eye_boxes  = list(eye_boxes  or [])
        self.update()

    # ── 内部几何 ──────────────────────────────────────────────────────

    def _fit_rect(self):
        if not self._pixmap: return None
        pw, ph, ww, wh = (self._pixmap.width(), self._pixmap.height(),
                          self.width(), self.height())
        if pw <= 0 or ph <= 0 or ww <= 0 or wh <= 0: return None
        s = min(ww / pw, wh / ph); dw, dh = pw * s, ph * s
        return QRectF((ww - dw) / 2, (wh - dh) / 2, dw, dh)

    def _100_rect(self):
        if not self._pixmap: return None
        return QRectF(float(self._offset.x()), float(self._offset.y()),
                      float(self._pixmap.width()), float(self._pixmap.height()))

    def _cur_rect(self):
        return self._fit_rect() if self._mode == 'fit' else self._100_rect()

    def _clamp(self):
        if not self._pixmap: return
        pw, ph, ww, wh = (self._pixmap.width(), self._pixmap.height(),
                          self.width(), self.height())
        M = 40; ox, oy = self._offset.x(), self._offset.y()
        ox = max(ww - pw + M, min(ox, -M)) if pw >= ww else (ww - pw) // 2
        oy = max(wh - ph + M, min(oy, -M)) if ph >= wh else (wh - ph) // 2
        self._offset = QPoint(int(ox), int(oy))

    def _primary_af(self):
        if not self._af_points or not self._pixmap: return None
        if self._af_ref_w <= 0 or self._af_ref_h <= 0: return None
        p = (next((x for x in self._af_points if x.in_focus), None) or
             next((x for x in self._af_points if x.selected), None))
        if not p: return None
        return (p.cx * self._pixmap.width()  / self._af_ref_w,
                p.cy * self._pixmap.height() / self._af_ref_h)

    # ── paintEvent ────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#0d0e12'))
        if not self._pixmap:
            p.setPen(QColor('#3a4050')); p.setFont(QFont(UI_FONT_FAMILY, 13))
            p.drawText(self.rect(), int(_ALIGN_CENTER),
                       '请通过工具栏打开 RAW 文件\n支持格式：.nef   .cr2   .cr3   .arw   .orf')
            return

        rect = self._cur_rect()
        if rect is None: return

        if self._mode == 'fit':
            if (self._cached_fit_pm is None or
                    self._cached_fit_pm.size() != rect.size().toSize()):
                self._cached_fit_pm = self._pixmap.scaled(
                    int(rect.width()), int(rect.height()), _KEEP_AR, _SMOOTH_XFORM)
            p.drawPixmap(int(rect.x()), int(rect.y()), self._cached_fit_pm)
        else:
            p.drawPixmap(int(rect.x()), int(rect.y()), self._pixmap)

        if self._flag == FLAG_REJECT:
            p.setRenderHint(_ANTIALIASING)
            border_pen = QPen(QColor(155, 68, 85, 160), 5)
            border_pen.setStyle(_SOLID_LINE)
            p.setPen(border_pen); p.setBrush(QBrush(_NO_BRUSH))
            p.drawRect(3, 3, self.width() - 6, self.height() - 6)

        self._draw_yolo_overlay(p)
        self._draw_af_overlay(p)
        self._draw_flag_badge(p)
        self._draw_sharpness_badge(p)
        self._draw_fused_badge(p)
        self._draw_exif_overlay(p)

    # ── 旗标徽章 ──────────────────────────────────────────────────────

    def _draw_flag_badge(self, p: QPainter):
        if self._flag == FLAG_NONE: return
        is_pick = (self._flag == FLAG_PICK)
        icon = '★ 选用' if is_pick else '✕ 待删除'
        bg   = CLR_PICK_BG if is_pick else CLR_REJECT_BG

        font = QFont(UI_FONT_FAMILY, 15); font.setBold(True)
        p.setFont(font); fm = p.fontMetrics()
        tw = fm.horizontalAdvance(icon)
        pad_x, pad_y = 14, 8
        bw = tw + pad_x * 2; bh = fm.height() + pad_y * 2
        bx, by = 14, 14

        p.setRenderHint(_ANTIALIASING)
        p.setBrush(QBrush(bg)); p.setPen(QPen(QColor(0, 0, 0, 0)))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 6, 6)
        p.setPen(QPen(CLR_PICK_FG))
        p.drawText(int(bx + pad_x), int(by + pad_y + fm.ascent()), icon)

    # ── 评分徽章（共用实现）──────────────────────────────────────────

    def _draw_score_badge(self, p: QPainter, lbl_txt: str, val_txt: str,
                          val_clr: QColor, by: float):
        """绘制双行评分徽章（标签行 + 数值行），左侧装饰色条。"""
        font_lbl = QFont(UI_FONT_FAMILY, 10)
        font_val = QFont(UI_FONT_FAMILY, 14); font_val.setBold(True)
        bg_alpha = 180

        p.setFont(font_lbl); fm_lbl = p.fontMetrics()
        lbl_w = fm_lbl.horizontalAdvance(lbl_txt); lbl_asc = fm_lbl.ascent(); lbl_h = fm_lbl.height()
        p.setFont(font_val); fm_val = p.fontMetrics()
        val_w = fm_val.horizontalAdvance(val_txt); val_asc = fm_val.ascent(); val_h = fm_val.height()

        inner_w = max(lbl_w, val_w); inner_h = lbl_h + 2 + val_h
        pad_x, pad_y = 12, 8
        bw = inner_w + pad_x * 2; bh = inner_h + pad_y * 2
        bx = (self.width() - bw) / 2

        p.setRenderHint(_ANTIALIASING)
        p.setBrush(QBrush(QColor(10, 11, 16, bg_alpha)))
        bpen = QPen(QColor(255, 255, 255, 20)); bpen.setWidthF(0.7)
        p.setPen(bpen); p.drawRoundedRect(QRectF(bx, by, bw, bh), 6, 6)

        barp = QPen(val_clr, 3.0); barp.setCapStyle(_ROUND_CAP)
        p.setPen(barp)
        p.drawLine(int(bx + 5), int(by + 8), int(bx + 5), int(by + bh - 8))

        txt_x = bx + pad_x + 3
        p.setFont(font_lbl); p.setPen(QPen(QColor(130, 140, 165, bg_alpha + 40)))
        p.drawText(int(txt_x), int(by + pad_y + lbl_asc), lbl_txt)
        p.setFont(font_val)
        vc = QColor(val_clr); vc.setAlpha(255); p.setPen(QPen(vc))
        p.drawText(int(txt_x), int(by + pad_y + lbl_h + 2 + val_asc), val_txt)

    def _draw_sharpness_badge(self, p: QPainter):
        if self._sharpness <= 0: return
        v = self._sharpness
        clr = (SCORE_CLR_GREEN if v >= SHARP_HI
               else (SCORE_CLR_YELLOW if v >= SHARP_LO else SCORE_CLR_RED))
        self._draw_score_badge(p, '基础分', str(int(v)), clr, by=100)

    def _draw_fused_badge(self, p: QPainter):
        if self._fused_score <= 0: return
        s = self._fused_score
        clr = (SCORE_CLR_GREEN if s >= 700
               else (SCORE_CLR_YELLOW if s >= 450 else SCORE_CLR_RED))
        self._draw_score_badge(p, '综合分', str(int(s)), clr, by=35)

    # ── YOLO 叠加层 ───────────────────────────────────────────────────

    def _draw_yolo_overlay(self, p: QPainter):
        if not self._pixmap: return
        if not self._show_yolo_bird and not self._show_yolo_eye: return
        if not (self._yolo_bird_boxes or self._yolo_bird_polys or self._yolo_eye_boxes): return
        rect = self._cur_rect()
        if rect is None: return
        pm_w = self._pixmap.width(); pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0: return
        sx = rect.width() / pm_w; sy = rect.height() / pm_h

        p.setRenderHint(_ANTIALIASING)

        if self._show_yolo_bird:
            bird_pen = QPen(QColor(70, 160, 255, 235))
            bird_pen.setWidthF(max(1.1, 2.4 * min(sx, sy)))
            bird_pen.setStyle(_SOLID_LINE)
            p.setPen(bird_pen)

            if self._yolo_bird_polys:
                # 优先画分割多边形
                for poly in self._yolo_bird_polys:
                    if not poly or len(poly) < 2: continue
                    prev = poly[-1]
                    for cur in poly:
                        p.drawLine(int(rect.x() + prev[0] * sx), int(rect.y() + prev[1] * sy),
                                   int(rect.x() + cur[0]  * sx), int(rect.y() + cur[1]  * sy))
                        prev = cur
            elif self._yolo_bird_boxes:
                # 分割掩码缺失时回退到 bounding box
                p.setBrush(QBrush(_NO_BRUSH))
                for (x1, y1, x2, y2, _conf) in self._yolo_bird_boxes:
                    p.drawRect(QRectF(
                        rect.x() + float(x1) * sx,
                        rect.y() + float(y1) * sy,
                        max(1.0, float(x2 - x1) * sx),
                        max(1.0, float(y2 - y1) * sy),
                    ))

        if self._yolo_eye_boxes and self._show_yolo_eye:
            pen = QPen(QColor(255, 80, 220, 240))
            pen.setWidthF(max(1.0, 2.2 * min(sx, sy))); pen.setStyle(_SOLID_LINE)
            p.setPen(pen); p.setBrush(QBrush(_NO_BRUSH))
            for (x1, y1, x2, y2, _conf) in self._yolo_eye_boxes:
                p.drawRect(QRectF(rect.x() + float(x1) * sx, rect.y() + float(y1) * sy,
                                  max(1.0, float(x2 - x1) * sx), max(1.0, float(y2 - y1) * sy)))

    # ── AF 叠加层 ─────────────────────────────────────────────────────

    def _draw_af_overlay(self, p: QPainter):
        if not self._show_af: return
        if not self._pixmap or not self._af_points: return
        if self._af_ref_w <= 0 or self._af_ref_h <= 0: return
        rect = self._cur_rect()
        if rect is None: return
        pm_w = self._pixmap.width(); pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0: return

        scale   = min(rect.width() / pm_w, rect.height() / pm_h)
        base_lw = max(2, min(6, int(min(pm_w, pm_h) * 0.003)))
        lw      = max(0.7, float(base_lw) * float(scale))
        sx      = pm_w / self._af_ref_w
        sy      = pm_h / self._af_ref_h

        p.setRenderHint(_ANTIALIASING)
        for afp in self._af_points:
            cx_pm = afp.cx * sx; cy_pm = afp.cy * sy
            bw_pm = max(afp.w * sx, 24.0); bh_pm = max(afp.h * sy, 24.0)
            cx = rect.x() + cx_pm * scale; cy = rect.y() + cy_pm * scale
            bw = bw_pm * scale;            bh = bh_pm * scale

            color = (CLR_IN_FOCUS if afp.in_focus
                     else (CLR_SELECTED if afp.selected else CLR_INACTIVE))
            pen = QPen(color); pen.setWidthF(lw); pen.setStyle(_SOLID_LINE)
            p.setPen(pen); p.setBrush(QBrush(_NO_BRUSH))
            p.drawRect(QRectF(cx - bw / 2, cy - bh / 2, bw, bh))

            cs_pm = max(8, int(min(bw_pm, bh_pm) * 0.18))
            cs = max(2, int(cs_pm * scale))
            p.drawLine(int(cx) - cs, int(cy), int(cx) + cs, int(cy))
            p.drawLine(int(cx), int(cy) - cs, int(cx), int(cy) + cs)

    # ── EXIF 叠加层 ───────────────────────────────────────────────────

    def _draw_exif_overlay(self, p: QPainter):
        if not self._show_exif or not self._exif_data: return
        rows = get_exif_overlay_rows(self._exif_data)
        if not rows: return

        W = self.width(); n_rows = len(rows)
        bar_h = n_rows * _EXIF_BAR_LINE_H + _EXIF_BAR_PAD_Y * 2
        bar_y = self.height() - bar_h

        font_lbl = QFont(UI_FONT_FAMILY, 11)
        font_val = QFont(UI_FONT_FAMILY, 11); font_val.setBold(True)
        clr_lbl = QColor( 78,  88, 108); clr_val = QColor(175, 185, 205)
        clr_sep = QColor( 48,  54,  70)
        fm_lbl  = QFontMetrics(font_lbl); fm_val  = QFontMetrics(font_val)

        max_x = float(_EXIF_BAR_PAD_X); max_w = float(W) - 16.0
        for row_fields in rows:
            x = float(_EXIF_BAR_PAD_X)
            for fi, (lbl, val) in enumerate(row_fields):
                if x > max_w - 60: break
                if fi > 0: x += fm_lbl.horizontalAdvance(' │ ')
                x += fm_lbl.horizontalAdvance(lbl + ' ')
                x += fm_val.horizontalAdvance(val) + _EXIF_COL_GAP
            if x > max_x: max_x = x

        bg_x = 8.0
        bg_w = min(max(240.0, max_x - bg_x + 4.0), float(W) - bg_x - 8.0)
        bg_top  = float(bar_y) - 16.0
        bg_rect = QRectF(bg_x, bg_top, bg_w, float(bar_h) + 16.0)

        p.setRenderHint(_ANTIALIASING)
        grad = QLinearGradient(0, bg_rect.top(), 0, bg_rect.bottom())
        grad.setColorAt(0.0, QColor(4, 5, 10, 0)); grad.setColorAt(0.40, QColor(4, 5, 10, 150))
        grad.setColorAt(1.0, QColor(4, 5, 10, 218))
        p.setBrush(QBrush(grad))
        bpen = QPen(QColor(255, 255, 255, 14)); bpen.setWidthF(0.8)
        p.setPen(bpen); p.drawRoundedRect(bg_rect, 8, 8)
        top_pen = QPen(QColor(255, 255, 255, 10)); top_pen.setWidthF(1.0)
        p.setPen(top_pen)
        p.drawLine(int(bg_x + 10), int(bar_y), int(bg_x + bg_w - 10), int(bar_y))

        base_y = bar_y + _EXIF_BAR_PAD_Y; right_limit = bg_x + bg_w - 10
        for row_i, row_fields in enumerate(rows):
            y = base_y + row_i * _EXIF_BAR_LINE_H + fm_val.ascent()
            x = _EXIF_BAR_PAD_X
            for fi, (lbl, val) in enumerate(row_fields):
                if x > right_limit - 60: break
                if fi > 0:
                    p.setFont(font_lbl); p.setPen(QPen(clr_sep))
                    p.drawText(x, y, ' │ '); x += fm_lbl.horizontalAdvance(' │ ')
                lbl_str = lbl + ' '
                p.setFont(font_lbl); p.setPen(QPen(clr_lbl))
                p.drawText(x, y, lbl_str); x += fm_lbl.horizontalAdvance(lbl_str)
                p.setFont(font_val); p.setPen(QPen(clr_val))
                p.drawText(x, y, val); x += fm_val.horizontalAdvance(val) + _EXIF_COL_GAP

    # ── 右键菜单 ──────────────────────────────────────────────────────

    _MENU_STYLE = (
        'QMenu { background:#2f2f2f; color:#bfbfbf; font-size:13px;'
        ' border:1px solid #3a3a3a; border-radius:6px; padding:4px 0; }'
        'QMenu::item { padding:6px 24px 6px 16px; }'
        'QMenu::item:selected { background:#444444; color:#f2f2f2; border-radius:3px; }'
        'QMenu::item:checked  { color:#e2e2e2; }'
        'QMenu::separator { height:1px; background:#3a3a3a; margin:3px 8px; }'
        'QMenu::item:disabled { color:#6f6f6f; }'
    )

    def _ctx_menu(self, pos):
        if not self._pixmap: return
        menu = QMenu(self); menu.setStyleSheet(self._MENU_STYLE)
        if self._ctx_fname_str:
            menu.addAction(self._ctx_fname_str).setEnabled(False); menu.addSeparator()
        zoom_lbl = '切换 100% 视图' if self._mode == 'fit' else '切换适合窗口'
        menu.addAction(zoom_lbl).triggered.connect(
            lambda: self._toggle_zoom(QPoint(self.width() // 2, self.height() // 2)))
        menu.addSeparator()
        a_pick   = menu.addAction('标记选用  ( ~ )')
        a_reject = menu.addAction('标记待删除  ( d )')
        a_clear  = menu.addAction('清除标记  ( u )')
        a_pick.triggered.connect(self.sig_flag_pick)
        a_reject.triggered.connect(self.sig_flag_reject)
        a_clear.triggered.connect(self._on_ctx_clear_flag)
        menu.addSeparator()
        menu.addAction('隐藏EXIF信息' if self._show_exif else '显示EXIF信息'
                       ).triggered.connect(self._toggle_exif_from_ctx)
        menu.addAction('隐藏对焦范围' if self._show_af else '显示对焦范围'
                       ).triggered.connect(self._toggle_af_from_ctx)
        menu.addAction('隐藏鸟身轮廓' if self._show_yolo_bird else '显示鸟身轮廓'
                       ).triggered.connect(self._toggle_yolo_bird_from_ctx)
        menu.addAction('隐藏鸟眼范围' if self._show_yolo_eye else '显示鸟眼范围'
                       ).triggered.connect(self._toggle_yolo_eye_from_ctx)
        menu.addSeparator()
        menu.addAction('在文件浏览器中显示').triggered.connect(self.sig_reveal_current)
        menu.addAction('重新评分选中照片').triggered.connect(self.sig_rescore_selected)
        menu.addAction('删除此文件').triggered.connect(self.sig_delete_current)
        (menu.exec if _PYSIDE6 else menu.exec_)(self.mapToGlobal(pos))

    def _on_ctx_clear_flag(self):
        mw = self.window()
        if hasattr(mw, '_set_flag_current'): mw._set_flag_current(FLAG_NONE)

    def _toggle_exif_from_ctx(self):
        self._show_exif = not self._show_exif; self.update()
        self.sig_exif_toggled.emit(self._show_exif)

    def _toggle_af_from_ctx(self):
        self._show_af = not self._show_af; self.update()
        self.sig_af_toggled.emit(self._show_af)

    def _toggle_yolo_bird_from_ctx(self):
        self._show_yolo_bird = not self._show_yolo_bird; self.update()
        self.sig_yolo_bird_toggled.emit(self._show_yolo_bird)

    def _toggle_yolo_eye_from_ctx(self):
        self._show_yolo_eye = not self._show_yolo_eye; self.update()
        self.sig_yolo_eye_toggled.emit(self._show_yolo_eye)

    # ── 鼠标 / 滚轮 / 键盘事件 ───────────────────────────────────────

    def mousePressEvent(self, e):
        pos = e.position().toPoint() if hasattr(e, 'position') else e.pos()
        if e.button() == _LMB:
            self._press_pos = pos; self._drag_on = False
            if self._mode == '100': self.setCursor(QCursor(_HAND_CUR))

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint() if hasattr(e, 'position') else e.pos()
        if self._press_pos is None or not (e.buttons() & _LMB): return
        delta = pos - self._press_pos
        if not self._drag_on and delta.manhattanLength() > 5: self._drag_on = True
        if self._drag_on and self._mode == '100':
            self._offset += delta; self._press_pos = pos; self._clamp(); self.update()

    def mouseReleaseEvent(self, e):
        pos = e.position().toPoint() if hasattr(e, 'position') else e.pos()
        if e.button() == _LMB:
            self.setCursor(QCursor(_ARROW_CUR))
            if not self._drag_on: self._toggle_zoom(pos)
            self._press_pos = None; self._drag_on = False

    def _toggle_zoom(self, click):
        if not self._pixmap: return
        if self._mode == 'fit':
            fit = self._fit_rect()
            if fit is None or fit.width() <= 0:
                self._mode = '100'; self.update(); self.sig_zoom_chg.emit(); return
            s  = fit.width() / self._pixmap.width()
            af = self._primary_af()
            ix = af[0] if af else (click.x() - fit.x()) / s
            iy = af[1] if af else (click.y() - fit.y()) / s
            self._offset = QPoint(int(self.width() / 2 - ix), int(self.height() / 2 - iy))
            self._clamp(); self._mode = '100'
        else:
            self._mode = 'fit'; self._offset = QPoint(0, 0)
        self.update(); self.sig_zoom_chg.emit()

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if dy: self.sig_wheel_steps.emit(int(dy))

    def keyPressEvent(self, e):
        k = e.key()
        if k == _K_LEFT:    self.sig_prev.emit()
        elif k == _K_RIGHT: self.sig_next.emit()
        elif k == _K_ESC and self._mode == '100':
            self._mode = 'fit'; self._offset = QPoint(0, 0); self.update(); self.sig_zoom_chg.emit()
        elif k in (Qt.Key.Key_QuoteLeft, Qt.Key.Key_AsciiTilde):
            self.sig_flag_pick.emit()
        elif k == Qt.Key.Key_D:
            self.sig_flag_reject.emit()
        elif k == Qt.Key.Key_U:
            mw = self.window()
            if hasattr(mw, '_set_flag_current'): mw._set_flag_current(FLAG_NONE)
        elif k in (_K_DEL, _K_BACK):
            self.sig_delete_current.emit()
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, _):
        self.update()
