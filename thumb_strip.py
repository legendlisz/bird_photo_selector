#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缩略图相关控件：ThumbLoaderThread、ThumbnailStrip、ThumbnailScrollBar。"""

import io
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import rawpy; HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    from PIL import Image as PILImage, ImageOps; HAS_PIL = True
except ImportError:
    HAS_PIL = False

from qt_compat import (
    QWidget, QMenu, QSlider, QToolButton, QHBoxLayout,
    QFont, QColor, QPainter, QPen, QBrush, QPoint,
    QRectF, QCursor, QPixmap,
    QThread, Signal, Qt,
    _PYSIDE6, _STRONG_FOCUS, _CTX_CUSTOM,
    _ANTIALIASING, _ALIGN_CENTER, _TXT_SINGLE, _SOLID_LINE, _NO_BRUSH,
    _KEEP_AR, _SMOOTH_XFORM,
    _LMB, _ARROW_CUR, _MOD_CTRL, _MOD_SHIFT,
    _K_DEL, _K_BACK, _K_ESC,
    _FMT_RGB888, _FMT_RGBA8888,
)
from app_constants import (
    SB_BG, SB_BORDER,
    CLR_PICK_STRIP, CLR_REJ_STRIP,
    SCORE_CLR_GREEN, SCORE_CLR_YELLOW, SCORE_CLR_RED,
    UI_FONT_FAMILY,
    _STRIP_H, _CELL_W, _THUMB_W, _THUMB_H,
    _CELL_PAD_TOP, _LABEL_H, _LABEL_GAP,
)
from utils import FLAG_NONE, FLAG_PICK, FLAG_REJECT, natural_key, ScoreCache, FlagCache, SHARP_HI, SHARP_LO
from raw_reader import RAWReader, pil_to_qimage


# ════════════════════════════════════════════════════════════════════
#  § 缩略图加载线程
# ════════════════════════════════════════════════════════════════════

class ThumbLoaderThread(QThread):
    sig_loaded = Signal(int, object)

    def __init__(self, items: List[tuple]):
        super().__init__()
        self._items = items; self._cancelled = False

    def cancel(self): self._cancelled = True

    @staticmethod
    def _arr_to_qimage(arr: np.ndarray):
        if arr.ndim != 3: return None
        h, w, c = arr.shape
        from qt_compat import QImage
        fmt = _FMT_RGB888 if c == 3 else (_FMT_RGBA8888 if c == 4 else None)
        if fmt is None: return None
        return QImage(arr.tobytes(), w, h, w * c, fmt).copy()

    def run(self):
        TW, TH = _THUMB_W * 2, _THUMB_H * 2
        for idx, path in self._items:
            if self._cancelled: return
            qi = None
            try:
                if HAS_RAWPY:
                    try:
                        with rawpy.imread(path) as raw:
                            t = raw.extract_thumb()
                            if t.format == rawpy.ThumbFormat.JPEG:
                                data = bytes(t.data)
                                if HAS_PIL:
                                    img = PILImage.open(io.BytesIO(data))
                                    img = ImageOps.exif_transpose(img)
                                    img.thumbnail((TW, TH), PILImage.LANCZOS)
                                    qi  = pil_to_qimage(img)
                                else:
                                    from qt_compat import QImage
                                    tmp = QImage()
                                    if tmp.loadFromData(data):
                                        tmp = tmp.scaled(TW, TH, _KEEP_AR, _SMOOTH_XFORM); qi = tmp
                            elif t.format == rawpy.ThumbFormat.BITMAP:
                                arr = np.array(t.data)
                                if arr.ndim == 3:
                                    if HAS_PIL:
                                        img = PILImage.fromarray(arr)
                                        img.thumbnail((TW, TH), PILImage.LANCZOS)
                                        qi  = pil_to_qimage(img)
                                    else:
                                        qi  = self._arr_to_qimage(arr)
                    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                        pass
                if qi is None and HAS_PIL:
                    img = PILImage.open(path); img = ImageOps.exif_transpose(img)
                    img.thumbnail((TW, TH), PILImage.LANCZOS); qi = pil_to_qimage(img)
                if qi is None:
                    tmp, _ = RAWReader.read_qimage(path)
                    if tmp and not tmp.isNull(): qi = tmp.scaled(TW, TH, _KEEP_AR, _SMOOTH_XFORM)
            except Exception as e:
                print(f"[ThumbLoader] {os.path.basename(path)}: {e}")
            if not self._cancelled: self.sig_loaded.emit(idx, qi)


# ════════════════════════════════════════════════════════════════════
#  § 缩略图条
# ════════════════════════════════════════════════════════════════════

class ThumbnailStrip(QWidget):
    sig_navigate                  = Signal(int)
    sig_delete_paths              = Signal(list)
    sig_visible_changed           = Signal()
    sig_scroll_changed            = Signal(int)
    sig_sort_selected             = Signal(object)
    sig_group_inner_sort_selected = Signal(str)
    sig_expand_burst_group        = Signal(int)

    _C_BG            = QColor(0x2b, 0x2b, 0x2b)
    _C_HVR           = QColor(0x34, 0x34, 0x34)
    _C_CUR_BG        = QColor(0xd8, 0xd8, 0xd8, 68)
    _C_SEL_BG        = QColor(0xa8, 0xa8, 0xa8, 34)
    _C_THUMB_CUR_BDR = QColor( 72, 199, 107, 255)
    _C_THUMB_SEL_BDR = QColor( 55, 168,  85, 210)
    _C_LABEL         = QColor(0x95, 0x95, 0x95)
    _C_LBL_CUR       = QColor(0xe0, 0xe0, 0xe0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_STRIP_H); self.setFocusPolicy(_STRONG_FOCUS)
        self.setMouseTracking(True); self.setContextMenuPolicy(_CTX_CUSTOM)
        self.customContextMenuRequested.connect(self._ctx_menu)
        self.setStyleSheet(f'background:{SB_BG};border-top:1px solid {SB_BORDER};')

        self._paths: List[str]                    = []
        self._thumbs: Dict[int, object]           = {}
        self._cached_thumbs: Dict[int, object]    = {}
        self._scores: Dict[int, Optional[float]]  = {}
        self._fused_scores: Dict[int, Optional[float]] = {}
        self._flags: Dict[int, int]               = {}
        self._current: int                        = -1
        self._selected: set                       = set()
        self._scroll_x: int                       = 0
        self._hover: int                          = -1
        self._loader: Optional[ThumbLoaderThread] = None
        self._filter_tiers: set                   = {'green', 'yellow', 'red'}
        self._filter_mode: str                    = 'fused_only'
        self._filter_flags: set                   = {'pick', 'none', 'reject'}
        self._burst_fold: bool                    = False
        self._show_burst_groups: bool             = True
        self._burst_label_head_only: bool         = True
        self._sort_by_burst_group: bool           = False
        self._burst_group_inner_sort_mode: str    = 'name'
        self._burst_groups: List[List[int]]       = []
        self._burst_map: Dict[int, int]           = {}
        self._burst_hidden: set                   = set()
        self._burst_group_owner: Dict[int, int]   = {}
        self._burst_group_label: Dict[int, int]   = {}
        self._burst_member_count: Dict[int, int]  = {}
        self._burst_idx_color_idx: Dict[int, int] = {}
        self._burst_badge_rects: Dict[int, QRectF]= {}
        self._bird_boxes: Dict[int, str]          = {}
        self._sort_order: Optional[str]           = None
        self._visible: List[int]                  = []
        self._visible_painted: List[int]          = []
        self._sel_anchor: int                     = -1
        self._sort_dirty: bool                    = False

    # ── 初始化/设置 ───────────────────────────────────────────────────

    def set_files(self, paths, score_cache=None, flag_cache=None, current_idx=0, start_loader=True):
        self._stop_loader()
        self._paths = list(paths)
        self._thumbs = {}; self._cached_thumbs = {}; self._scores = {}
        self._fused_scores = {}; self._flags = {}; self._bird_boxes = {}
        self._selected = set(); self._current = -1; self._sel_anchor = -1
        self._visible_painted = []; self._sort_dirty = False; self._burst_idx_color_idx = {}
        if score_cache:
            for i, p in enumerate(self._paths):
                v = score_cache.get(p)
                if v is not None: self._scores[i] = v
                fv = score_cache.get_fused(p)
                if fv is not None: self._fused_scores[i] = fv
                bx = score_cache.get_bird_box(p)
                if bx: self._bird_boxes[i] = bx
        if flag_cache:
            for i, p in enumerate(self._paths):
                f = flag_cache.get(p)
                if f != FLAG_NONE: self._flags[i] = f
        self._compute_burst_groups(); self._recompute_burst_best()
        self._rebuild_visible(); self.set_scroll_x(self._scroll_x)
        if start_loader: self._start_loader(priority_idx=current_idx)
        self.update()

    def set_burst_fold(self, enabled):
        self._burst_fold = bool(enabled)
        if enabled: self._recompute_burst_best()
        self._rebuild_visible(); self.update()

    def set_show_burst_groups(self, enabled):
        self._show_burst_groups = bool(enabled); self.update()

    def set_burst_label_head_only(self, enabled):
        self._burst_label_head_only = bool(enabled); self.update()

    def set_sort_by_burst_group(self, enabled):
        self._sort_by_burst_group = bool(enabled); self._sort_dirty = False
        self._rebuild_visible(); self._scroll_to(self._current); self.update()

    def set_burst_group_inner_sort_mode(self, mode):
        mode = (mode or 'name').strip().lower()
        if mode not in ('name', 'sharp', 'fused'): mode = 'name'
        self._burst_group_inner_sort_mode = mode; self._sort_dirty = False
        self._rebuild_visible(); self._scroll_to(self._current); self.update()

    # ── 连拍分组 ──────────────────────────────────────────────────────

    def _compute_burst_groups(self):
        self._burst_groups = []
        if not self._paths: return
        sorted_indices = sorted(range(len(self._paths)),
            key=lambda i: natural_key(os.path.basename(self._paths[i])))
        current_group = []; last_mtime = 0.0; last_box = None

        def _parse_box(s):
            if not s: return None
            try: return [int(x) for x in s.split(',')]
            except Exception: return None

        for i in sorted_indices:
            p = self._paths[i]
            try: mtime = os.path.getmtime(p)
            except Exception: mtime = 0.0
            box = _parse_box(self._bird_boxes.get(i))
            if not current_group:
                current_group.append(i); last_mtime = mtime; last_box = box
            else:
                time_ok = abs(mtime - last_mtime) <= 1.8; pos_ok = True
                if box and last_box:
                    cx1 = (box[0]+box[2])/2; cy1 = (box[1]+box[3])/2
                    cx2 = (last_box[0]+last_box[2])/2; cy2 = (last_box[1]+last_box[3])/2
                    w1,h1 = max(1,box[2]-box[0]), max(1,box[3]-box[1])
                    w2,h2 = max(1,last_box[2]-last_box[0]), max(1,last_box[3]-last_box[1])
                    dist_sq = (cx1-cx2)**2+(cy1-cy2)**2
                    max_dist = max(w1,w2,h1,h2)*2.2
                    a1,a2 = w1*h1, w2*h2
                    area_ratio = (a1/a2) if a1>=a2 else (a2/a1)
                    if dist_sq > max_dist**2 and area_ratio > 3.0: pos_ok = False
                if time_ok and pos_ok:
                    current_group.append(i); last_mtime = mtime; last_box = box or last_box
                else:
                    if len(current_group)>1: self._burst_groups.append(current_group)
                    current_group=[i]; last_mtime=mtime; last_box=box
        if len(current_group)>1: self._burst_groups.append(current_group)

    def _burst_outer_score(self, idx):
        order = self._sort_order
        if order in ('fdesc','fasc'): return self._fused_scores.get(idx)
        if order in ('desc','asc'):   return self._scores.get(idx)
        s = self._fused_scores.get(idx)
        if s is None or s<=0: s = self._scores.get(idx)
        return s

    def _recompute_burst_best(self):
        self._burst_map.clear(); self._burst_hidden.clear()
        self._burst_group_owner.clear(); self._burst_group_label.clear(); self._burst_member_count.clear()
        for gid, g in enumerate(self._burst_groups, start=1):
            best_i=g[0]; best_score=float('-inf')
            for i in g:
                s=self._burst_outer_score(i); sval=float(s) if s is not None else float('-inf')
                if sval>best_score: best_score=sval; best_i=i
            self._burst_map[best_i]=len(g)
            for i in g:
                self._burst_group_owner[i]=best_i; self._burst_group_label[i]=gid
                self._burst_member_count[i]=len(g)
                if i!=best_i: self._burst_hidden.add(i)

    def get_burst_best_for(self, idx):
        if idx<0: return idx
        for g in self._burst_groups:
            if idx in g:
                best_i=g[0]; best_score=float('-inf')
                for i in g:
                    s=self._burst_outer_score(i); sval=float(s) if s is not None else float('-inf')
                    if sval>best_score: best_score=sval; best_i=i
                return best_i
        return idx

    # ── 旗标 / 分数更新 ───────────────────────────────────────────────

    def set_flags_bulk(self, idx_to_flag):
        if not idx_to_flag: return
        changed=False
        for idx, flag in idx_to_flag.items():
            if idx<0 or idx>=len(self._paths): continue
            if flag==FLAG_NONE:
                if idx in self._flags: self._flags.pop(idx,None); changed=True
            else:
                if self._flags.get(idx)!=flag: self._flags[idx]=flag; changed=True
        if not changed: return
        self._rebuild_visible(); self.update()

    def set_flag(self, idx, flag): self.set_flags_bulk({idx: flag})

    def set_current(self, idx):
        if self._sort_dirty and self._sort_order in ('asc','desc'):
            self._rebuild_visible(); self._sort_dirty=False
        self._current=idx; self._sel_anchor=idx; self._selected={idx}
        self._scroll_to(idx); self._stop_loader(); self._start_loader(priority_idx=idx); self.update()

    def update_score(self, idx, score, source='live'):
        self._scores[idx]=score
        if self._burst_fold: self._recompute_burst_best()
        if self._sort_order in ('asc','desc') or self._burst_fold:
            self._sort_dirty=True
            if idx!=self._current:
                self._rebuild_visible(); self._sort_dirty=False; self.set_scroll_x(self._scroll_x)
        self.update()

    def update_fused_score(self, idx, score, source='live'):
        self._fused_scores[idx]=score
        if self._burst_fold: self._recompute_burst_best()
        if self._sort_order in ('fasc','fdesc') or self._burst_fold:
            self._sort_dirty=True
            if idx!=self._current:
                self._rebuild_visible(); self._sort_dirty=False; self.set_scroll_x(self._scroll_x)
        self.update()

    def stop_loader(self): self._stop_loader()

    # ── 过滤 / 排序 ───────────────────────────────────────────────────

    def _score_tier(self, score, is_fused=False):
        if is_fused:
            if score>=700: return 'green'
            if score>=450: return 'yellow'
            return 'red'
        if score>=SHARP_HI: return 'green'
        if score>=SHARP_LO: return 'yellow'
        return 'red'

    def _matches_filter(self, idx):
        if self._burst_fold and idx in self._burst_hidden: return False
        sharp=self._scores.get(idx); fused=self._fused_scores.get(idx)
        has_sharp=sharp is not None and float(sharp)>0
        has_fused=fused is not None and float(fused)>0
        sharp_ok=(not has_sharp) or (self._score_tier(sharp,False) in self._filter_tiers)
        fused_ok=(not has_fused) or (self._score_tier(fused,True)  in self._filter_tiers)
        mode=self._filter_mode
        if mode=='off': pass
        elif mode=='sharp_only':
            if has_sharp and not sharp_ok: return False
            if not has_sharp and has_fused: return False
        elif mode=='fused_only':
            if has_fused and not fused_ok: return False
            if not has_fused and has_sharp: return False
        elif mode=='both':
            if not (has_sharp and has_fused): return False
            if not (sharp_ok and fused_ok): return False
        if self._filter_flags != {'pick','none','reject'}:
            flag=self._flags.get(idx, FLAG_NONE)
            flag_key={FLAG_PICK:'pick',FLAG_NONE:'none',FLAG_REJECT:'reject'}.get(flag,'none')
            if flag_key not in self._filter_flags: return False
        return True

    def _rebuild_visible(self):
        vis=[i for i in range(len(self._paths)) if self._matches_filter(i)]
        order=self._sort_order

        def _score_of(i): return self._fused_scores.get(i) if order in ('fdesc','fasc') else self._scores.get(i)
        def _item_key(i):
            sc=_score_of(i); miss=1 if sc is None else 0; sval=float(sc) if sc is not None else 0.0
            nkey=natural_key(os.path.basename(self._paths[i]))
            return (miss,sval,nkey) if order in ('asc','fasc') else (miss,-sval,nkey)

        use_group_sort=(self._show_burst_groups or self._burst_fold or self._sort_by_burst_group)
        if use_group_sort:
            owner_groups={}; singles=[]
            for i in vis:
                if self._burst_member_count.get(i,0)>1:
                    owner=self._burst_group_owner.get(i,i); owner_groups.setdefault(owner,[]).append(i)
                else: singles.append([i])
            grouped_items=list(owner_groups.values())+singles

            def _group_key(members):
                if order in ('desc','asc','fdesc','fasc'):
                    vals=[float(v) for v in (_score_of(j) for j in members) if v is not None]
                    miss=1 if not vals else 0; anchor=max(vals) if vals else 0.0
                    nkey=natural_key(os.path.basename(self._paths[min(members)]))
                    return (miss,anchor,nkey) if order in ('asc','fasc') else (miss,-anchor,nkey)
                return (0, natural_key(os.path.basename(self._paths[min(members)])))

            grouped_items.sort(key=_group_key); vis=[]
            for members in grouped_items:
                m=self._burst_group_inner_sort_mode
                if m=='name':
                    members.sort(key=lambda j: natural_key(os.path.basename(self._paths[j])))
                else:
                    src=self._scores if m=='sharp' else self._fused_scores
                    desc_like=order not in ('asc','fasc')
                    def _inner_key(j):
                        sc=src.get(j); miss=1 if sc is None else 0; sval=float(sc) if sc is not None else 0.0
                        nkey=natural_key(os.path.basename(self._paths[j]))
                        return (miss,-sval,nkey) if desc_like else (miss,sval,nkey)
                    members.sort(key=_inner_key)
                vis.extend(members)
        elif order in ('desc','asc','fdesc','fasc'): vis.sort(key=_item_key)
        self._visible=vis; self.sig_visible_changed.emit(); self.sig_scroll_changed.emit(self._scroll_x)

    # ── 滚动 ──────────────────────────────────────────────────────────

    def max_scroll(self): return max(0, len(self._visible)*_CELL_W-self.width())
    def scroll_x(self):   return int(self._scroll_x)

    def set_scroll_x(self, x):
        x=max(0,min(int(x),self.max_scroll()))
        if x==self._scroll_x: self.sig_scroll_changed.emit(self._scroll_x); return
        self._scroll_x=x; self._reprioritize_loader_for_viewport()
        self.update(); self.sig_scroll_changed.emit(self._scroll_x)

    def _scroll_to(self, idx):
        if idx<0: return
        try: vpos=self._visible.index(idx)
        except ValueError: return
        item_left=vpos*_CELL_W; item_right=item_left+_CELL_W
        if item_left>=self._scroll_x and item_right<=self._scroll_x+self.width(): return
        self.set_scroll_x(item_left+_CELL_W//2-self.width()//2)

    def _idx_at(self, wx):
        vis=self._visible_painted if self._visible_painted else self._visible
        col=(wx+self._scroll_x)//_CELL_W
        return vis[col] if 0<=col<len(vis) else -1

    # ── 加载线程管理 ──────────────────────────────────────────────────

    def _stop_loader(self):
        from workers import retire_thread
        retire_thread(self._loader); self._loader=None

    def _start_loader(self, priority_idx=0):
        if not self._paths: return
        self._stop_loader()
        try:    vpos_cur=self._visible.index(priority_idx)
        except ValueError: vpos_cur=0
        pos_map={idx:pos for pos,idx in enumerate(self._visible)}
        def _priority(item):
            i,_=item; pos=pos_map.get(i)
            return abs(pos-vpos_cur) if pos is not None else len(self._visible)+i
        pending=[(i,p) for i,p in enumerate(self._paths) if self._thumbs.get(i) is None]
        if not pending: return
        items=sorted(pending,key=_priority)
        t=ThumbLoaderThread(items); t.sig_loaded.connect(self._on_thumb_loaded)
        self._loader=t; t.start()
        try: t.setPriority(QThread.Priority.LowPriority if _PYSIDE6 else QThread.LowPriority)
        except Exception: pass

    def _reprioritize_loader_for_viewport(self):
        if not self._visible: return
        first=max(0,self._scroll_x//_CELL_W)
        last=min(len(self._visible)-1,(self._scroll_x+self.width())//_CELL_W+1)
        if first>last: return
        needs_load=any(self._thumbs.get(self._visible[vi]) is None for vi in range(first,last+1))
        if not needs_load: return
        center_vi=(first+last)//2
        center_idx=self._visible[center_vi] if 0<=center_vi<len(self._visible) else self._current
        self._stop_loader(); self._start_loader(priority_idx=center_idx)

    def _on_thumb_loaded(self, idx, qi):
        if qi is not None and not qi.isNull():
            pm=QPixmap.fromImage(qi)
            self._thumbs[idx]=pm
            self._cached_thumbs[idx]=pm.scaled(_THUMB_W,_THUMB_H,_KEEP_AR,_SMOOTH_XFORM)
        else:
            self._thumbs[idx]=None; self._cached_thumbs[idx]=None
        self.update()
        try:
            vpos=self._visible.index(idx); cx=vpos*_CELL_W-self._scroll_x
            if 0<=cx+_CELL_W and cx<self.width(): self.update(cx,0,_CELL_W,self.height())
        except ValueError: pass

    # ── paintEvent ────────────────────────────────────────────────────

    def _rebuild_burst_color_map(self):
        self._burst_idx_color_idx={}
        if self._burst_fold or not self._show_burst_groups: return
        vis=self._visible_painted if self._visible_painted else self._visible
        last_owner=None; last_color=-1; current_color=0
        for idx in vis:
            if self._burst_member_count.get(idx,0)<=1: continue
            owner=self._burst_group_owner.get(idx,idx)
            if owner!=last_owner:
                if last_color>=0: current_color=1-last_color
                last_owner=owner; last_color=current_color
            self._burst_idx_color_idx[idx]=current_color

    def paintEvent(self, _):
        self._visible_painted=list(self._visible); self._burst_badge_rects.clear()
        self._rebuild_burst_color_map()
        p=QPainter(self); p.setRenderHint(_ANTIALIASING); p.fillRect(self.rect(),self._C_BG)
        if not self._visible:
            p.setPen(QColor('#3a4050')); p.setFont(QFont(UI_FONT_FAMILY,11))
            p.drawText(self.rect(),int(_ALIGN_CENTER),
                '请打开文件夹以显示缩略图' if not self._paths else '无符合过滤条件的照片')
            return
        first=max(0,self._scroll_x//_CELL_W)
        last=min(len(self._visible)-1,(self._scroll_x+self.width())//_CELL_W+1)
        for vi in range(first,last+1):
            idx=self._visible[vi]; self._draw_cell(p,idx,vi*_CELL_W-self._scroll_x)

    def _draw_cell(self, p, idx, cx):
        ch=self.height(); is_cur=(idx==self._current)
        is_sel=(idx in self._selected and not is_cur)
        is_hvr=(idx==self._hover and not is_cur and not is_sel)
        flag=self._flags.get(idx,FLAG_NONE)
        if is_cur:
            if flag==FLAG_REJECT: p.fillRect(cx,0,_CELL_W,ch,QColor(0x28,0x10,0x16))
            else:                  p.fillRect(cx,0,_CELL_W,ch,self._C_CUR_BG)
        elif is_sel: p.fillRect(cx,0,_CELL_W,ch,self._C_SEL_BG)
        elif is_hvr: p.fillRect(cx,0,_CELL_W,ch,self._C_HVR)

        group_color=None; group_label=0
        if (not self._burst_fold) and self._show_burst_groups:
            gcnt=self._burst_member_count.get(idx,0)
            if gcnt>1:
                cidx=self._burst_idx_color_idx.get(idx,0)
                group_color=QColor(176,176,176) if cidx==0 else QColor(146,146,146)
                group_label=self._burst_group_label.get(idx,0)

        tx=cx+(_CELL_W-_THUMB_W)//2; ty=_CELL_PAD_TOP
        pm=self._cached_thumbs.get(idx)
        if pm and pm.width()>0:
            ox=tx+(_THUMB_W-pm.width())//2; oy=ty+(_THUMB_H-pm.height())//2
            p.drawPixmap(ox,oy,pm)
            actual_l=ox; actual_r=ox+pm.width(); actual_t=oy
        else:
            p.fillRect(tx,ty,_THUMB_W,_THUMB_H,QColor(0x14,0x16,0x1e))
            p.setPen(QColor('#3a4050')); p.setFont(QFont(UI_FONT_FAMILY,9))
            p.drawText(QRectF(tx,ty,_THUMB_W,_THUMB_H),int(_ALIGN_CENTER),'…')
            actual_l=tx; actual_r=tx+_THUMB_W; actual_t=ty

        if is_cur or is_sel:
            _bdr_clr=self._C_THUMB_CUR_BDR if is_cur else self._C_THUMB_SEL_BDR
            _bdr_w=2.5 if is_cur else 2.0; _half=_bdr_w/2.0
            _ph=pm.height() if (pm and pm.width()>0) else _THUMB_H
            _bdr_pen=QPen(_bdr_clr,_bdr_w); _bdr_pen.setStyle(_SOLID_LINE)
            p.setPen(_bdr_pen); p.setBrush(QBrush(_NO_BRUSH))
            p.drawRect(QRectF(actual_l-_half,actual_t-_half,actual_r-actual_l+_bdr_w,_ph+_bdr_w))

        f=self._fused_scores.get(idx); s=self._scores.get(idx)
        burst_count=self._burst_map.get(idx,0) if self._burst_fold else 0
        show_fused=(f is not None and f>0); show_sharp=(s is not None and s>0)
        if self._burst_fold and burst_count>1:
            mode=self._sort_order
            if mode in ('desc','asc'): show_fused=False; show_sharp=(s is not None and s>0)
            else:                      show_fused=(f is not None and f>0); show_sharp=False

        if show_fused: self._draw_badge(p,actual_r,actual_t,int(f),is_fused=True)
        if burst_count>1:
            self._burst_badge_rects[idx]=self._draw_burst_badge(p,idx,actual_l,int(actual_t),burst_count)
        if show_sharp:
            actual_b=(actual_t+pm.height()) if (pm and pm.width()>0) else (ty+_THUMB_H)
            self._draw_badge(p,actual_r,int(actual_b)-15,int(s),is_fused=False)
        if flag!=FLAG_NONE: self._draw_flag_corner(p,tx,ty,flag)

        if group_color is not None:
            vis=self._visible_painted if self._visible_painted else self._visible
            try:   pos2=vis.index(idx)
            except ValueError: pos2=-1
            same_left=same_right=False
            if pos2>=0:
                owner=self._burst_group_owner.get(idx,idx)
                if pos2>0:
                    li=vis[pos2-1]
                    same_left=(self._burst_member_count.get(li,0)>1 and self._burst_group_owner.get(li,li)==owner)
                if pos2<len(vis)-1:
                    ri=vis[pos2+1]
                    same_right=(self._burst_member_count.get(ri,0)>1 and self._burst_group_owner.get(ri,ri)==owner)
            glow=QColor(group_color); glow.setAlpha(120); main=QColor(group_color); main.setAlpha(245)
            p.setPen(QPen(glow,3.0))
            p.drawLine(cx+1,1,cx+_CELL_W-2,1); p.drawLine(cx+1,ch-2,cx+_CELL_W-2,ch-2)
            if not same_left:  p.drawLine(cx+1,1,cx+1,ch-2)
            if not same_right: p.drawLine(cx+_CELL_W-2,1,cx+_CELL_W-2,ch-2)
            p.setPen(QPen(main,1.8))
            p.drawLine(cx+1,1,cx+_CELL_W-2,1); p.drawLine(cx+1,ch-2,cx+_CELL_W-2,ch-2)
            if not same_left:  p.drawLine(cx+1,1,cx+1,ch-2)
            if not same_right: p.drawLine(cx+_CELL_W-2,1,cx+_CELL_W-2,ch-2)
            draw_group_label=(not self._burst_label_head_only) or (not same_left)
            if draw_group_label:
                gtxt=f'组{group_label}'; gfont=QFont(UI_FONT_FAMILY,8); gfont.setBold(True); p.setFont(gfont)
                gfm=p.fontMetrics(); gtw=gfm.horizontalAdvance(gtxt)
                gbw,gbh=gtw+10,14; gbx,gby=cx+8,2
                p.setBrush(QBrush(QColor(12,14,20,230))); p.setPen(QPen(main,1.2))
                p.drawRoundedRect(QRectF(gbx,gby,gbw,gbh),4,4)
                p.setPen(QPen(main)); p.drawText(QRectF(gbx,gby,gbw,gbh),int(_ALIGN_CENTER),gtxt)

        name=os.path.splitext(os.path.basename(self._paths[idx]))[0]
        ly=ty+_THUMB_H+_LABEL_GAP; p.setFont(QFont(UI_FONT_FAMILY,9))
        lbl_clr=(QColor(0xb0,0x60,0x70) if flag==FLAG_REJECT else (self._C_LBL_CUR if is_cur else self._C_LABEL))
        p.setPen(lbl_clr)
        p.drawText(QRectF(cx+2,ly,_CELL_W-4,_LABEL_H),int(_ALIGN_CENTER)|int(_TXT_SINGLE),name)

    # ── 徽章助手 ──────────────────────────────────────────────────────

    def _draw_flag_corner(self, p, tx, ty, flag):
        size=18
        if flag==FLAG_PICK: clr=CLR_PICK_STRIP; txt='★'
        else:                clr=CLR_REJ_STRIP;  txt='✕'
        by=ty+_THUMB_H-size-1
        p.setBrush(QBrush(clr)); p.setPen(QPen(QColor(0,0,0,0)))
        p.drawRoundedRect(QRectF(tx,by,size,size),3,3)
        font=QFont(UI_FONT_FAMILY,9); font.setBold(True); p.setFont(font)
        p.setPen(QPen(QColor(255,255,255)))
        p.drawText(QRectF(tx,by,size,size),int(_ALIGN_CENTER),txt)

    def _draw_burst_badge(self, p, idx, left_x, top_y, count):
        txt=f'{max(1,int(count))}'; font=QFont(UI_FONT_FAMILY,8); font.setBold(True)
        p.setFont(font); fm=p.fontMetrics(); tw=fm.horizontalAdvance(txt)
        icon_sz=8; icon_w=icon_sz+1; bw=max(tw+icon_w+6,24); bh=14
        bx=left_x+2; by=top_y+2; rect=QRectF(bx,by,bw,bh)
        p.setBrush(QBrush(QColor(24,24,24,176))); p.setPen(QPen(QColor(96,96,96,190),0.7))
        p.drawRoundedRect(rect,3,3)
        p.setPen(QPen(QColor(160,160,160,220),1.0))
        p.drawRect(QRectF(bx+2,by+4,icon_sz-1,icon_sz-1)); p.drawRect(QRectF(bx+3,by+3,icon_sz-1,icon_sz-1))
        p.setPen(QPen(QColor(224,224,224,232)))
        p.drawText(QRectF(bx+icon_w,by,bw-icon_w,bh),int(_ALIGN_CENTER),txt)
        return rect

    def _draw_badge(self, p, right_x, top_y, score, is_fused=False):
        if is_fused:
            clr=SCORE_CLR_GREEN if score>=700 else (SCORE_CLR_YELLOW if score>=450 else SCORE_CLR_RED)
            txt=str(score) if score<10000 else f'{score//1000}k'
        else:
            clr=(SCORE_CLR_GREEN if score>=SHARP_HI else (SCORE_CLR_YELLOW if score>=SHARP_LO else SCORE_CLR_RED))
            txt=str(int(score))
        font=QFont(UI_FONT_FAMILY,8); font.setBold(True); p.setFont(font)
        fm=p.fontMetrics(); tw=fm.horizontalAdvance(txt); bw=max(tw+7,22); bh=14
        bx=right_x-bw-1; by=top_y+1
        p.setBrush(QBrush(QColor(0,0,0,160))); p.setPen(QPen(clr,0.6))
        p.drawRoundedRect(QRectF(bx,by,bw,bh),3,3)
        p.setPen(QPen(clr)); p.drawText(QRectF(bx,by,bw,bh),int(_ALIGN_CENTER),txt)

    # ── 事件 ──────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        pos=e.position().toPoint() if hasattr(e,'position') else e.pos()
        idx=self._idx_at(pos.x())
        if idx<0: return
        self.setFocus(); ctrl=bool(e.modifiers()&_MOD_CTRL); shift=bool(e.modifiers()&_MOD_SHIFT)
        if e.button()==_LMB:
            if self._burst_fold:
                brect=self._burst_badge_rects.get(idx)
                if brect is not None and brect.contains(pos): self.sig_expand_burst_group.emit(idx); return
            if ctrl:
                if idx in self._selected: self._selected.discard(idx)
                else:                      self._selected.add(idx)
                self._sel_anchor=idx
            elif shift and self._sel_anchor>=0:
                vis=self._visible_painted if self._visible_painted else self._visible
                try:
                    p1=vis.index(self._sel_anchor); p2=vis.index(idx)
                    lo,hi=min(p1,p2),max(p1,p2); self._selected=set(vis[lo:hi+1])
                except ValueError:
                    lo,hi=min(self._sel_anchor,idx),max(self._sel_anchor,idx)
                    self._selected=set(range(lo,hi+1))
            else:
                self._selected={idx}; self._sel_anchor=idx; self.sig_navigate.emit(idx)
        self.update()

    def mouseMoveEvent(self, e):
        pos=e.position().toPoint() if hasattr(e,'position') else e.pos()
        idx=self._idx_at(pos.x())
        if idx!=self._hover: self._hover=idx; self.update()

    def leaveEvent(self, _): self._hover=-1; self.update()

    def wheelEvent(self, e):
        dx=e.angleDelta().x(); dy=e.angleDelta().y()
        delta=dx if abs(dx)>abs(dy) else -dy
        self.set_scroll_x(self._scroll_x+delta//2)

    def keyPressEvent(self, e):
        k=e.key()
        if k in (_K_DEL,_K_BACK):
            if self._selected:
                paths=[self._paths[i] for i in sorted(self._selected) if 0<=i<len(self._paths)]
                if paths: self.sig_delete_paths.emit(paths)
        elif k==_K_ESC:
            self._selected={self._current} if self._current>=0 else set(); self.update()
        else: super().keyPressEvent(e)

    def resizeEvent(self, _):
        self._scroll_to(self._current); self.update(); self.sig_scroll_changed.emit(self._scroll_x)

    _MENU_STYLE=(
        'QMenu { background:#2f2f2f; color:#bfbfbf; font-size:13px;'
        ' border:1px solid #3a3a3a; border-radius:6px; padding:4px 0; }'
        'QMenu::item { padding:6px 24px 6px 16px; }'
        'QMenu::item:selected { background:#444444; color:#f2f2f2; border-radius:3px; }'
        'QMenu::item:checked  { color:#e2e2e2; }'
        'QMenu::separator { height:1px; background:#3a3a3a; margin:3px 8px; }'
        'QMenu::item:disabled { color:#6f6f6f; }'
    )

    def _ctx_menu(self, pos):
        idx=self._idx_at(pos.x())
        if idx>=0 and idx not in self._selected: self._selected={idx}; self.update()
        n=len(self._selected); menu=QMenu(self); menu.setStyleSheet(self._MENU_STYLE)
        if n==1:
            only=next(iter(self._selected))
            if 0<=only<len(self._paths):
                menu.addAction(os.path.basename(self._paths[only])).setEnabled(False); menu.addSeparator()
        act_first=menu.addAction('跳转到第一张'); act_last=menu.addAction('跳转到最后一张')
        has_vis=bool(self._visible)
        act_first.setEnabled(has_vis); act_last.setEnabled(has_vis)
        act_first.triggered.connect(lambda: self.sig_navigate.emit(self._visible[0]))
        act_last .triggered.connect(lambda: self.sig_navigate.emit(self._visible[-1]))
        menu.addSeparator()
        if n>0:
            menu.addAction(f'删除选中（{n} 个文件）').triggered.connect(self._emit_delete); menu.addSeparator()
        h=menu.addAction('—— 评分过滤 ——'); h.setEnabled(False)
        for mode, label in (('off','取消过滤'),('sharp_only','过滤基础分'),('fused_only','过滤综合分'),('both','双过滤')):
            a=menu.addAction(label); a.setCheckable(True); a.setChecked(self._filter_mode==mode)
            a.triggered.connect(lambda checked, m=mode: checked and self._set_filter_mode(m))
        for tier, label in (('green','绿色'),('yellow','黄色'),('red','红色')):
            a=menu.addAction(label); a.setCheckable(True); a.setChecked(tier in self._filter_tiers)
            a.triggered.connect(lambda checked, t=tier: self._toggle_filter(t,checked))
        menu.addAction('全部颜色').triggered.connect(self._show_all_tiers)
        menu.addSeparator()
        sm=menu.addMenu('排序')
        for key, label in ((None,'文件名顺序'),('desc','基础分从高到低'),('fdesc','综合得分从高到低')):
            a=sm.addAction(label); a.setCheckable(True); a.setChecked(self._sort_order==key)
            a.triggered.connect(lambda checked, k=key: self.sig_sort_selected.emit(k))
        im=menu.addMenu('组内排序')
        for key, label in (('name','文件名顺序'),('sharp','基础分从高到低'),('fused','综合得分从高到低')):
            a=im.addAction(label); a.setCheckable(True); a.setChecked(self._burst_group_inner_sort_mode==key)
            a.triggered.connect(lambda checked, k=key: checked and self.sig_group_inner_sort_selected.emit(k))
        (menu.exec if _PYSIDE6 else menu.exec_)(self.mapToGlobal(pos))

    def _set_filter_mode(self, mode):
        mode=(mode or 'fused_only').strip().lower()
        if mode not in ('off','sharp_only','fused_only','both'): mode='fused_only'
        self._filter_mode=mode
        if mode=='off': self._filter_tiers.update({'green','yellow','red'})
        self._rebuild_visible(); self._scroll_to(self._current); self.update()

    def _toggle_filter(self, tier, checked):
        if checked: self._filter_tiers.add(tier)
        else:       self._filter_tiers.discard(tier)
        self._rebuild_visible(); self._scroll_to(self._current); self.update()

    def _show_all_tiers(self):
        for t in ('green','yellow','red'): self._filter_tiers.add(t)
        self._rebuild_visible(); self._scroll_to(self._current); self.update()

    def _set_sort(self, order):
        self._sort_order=order; self._sort_dirty=False
        if self._burst_fold: self._recompute_burst_best()
        self._rebuild_visible(); self._scroll_to(self._current)
        self._stop_loader(); self._start_loader(priority_idx=self._current); self.update()

    def ensure_sorted(self):
        if self._sort_dirty and self._sort_order in ('asc','desc','fasc','fdesc'):
            self._rebuild_visible(); self._sort_dirty=False; self.set_scroll_x(self._scroll_x); self.update()

    def _emit_delete(self):
        paths=[self._paths[i] for i in sorted(self._selected) if 0<=i<len(self._paths)]
        if paths: self.sig_delete_paths.emit(paths)


# ════════════════════════════════════════════════════════════════════
#  § 缩略图滚动条
# ════════════════════════════════════════════════════════════════════

class ThumbnailScrollBar(QWidget):
    def __init__(self, strip: ThumbnailStrip, parent=None):
        super().__init__(parent)
        self._strip=strip; self.setFixedHeight(22)
        self.setStyleSheet(f'background:{SB_BG};')
        lay=QHBoxLayout(self); lay.setContentsMargins(8,3,8,3); lay.setSpacing(8)
        self._btn_first=QToolButton(self); self._btn_first.setText('|<')
        self._btn_last =QToolButton(self); self._btn_last .setText('>|')
        for b in (self._btn_first,self._btn_last): b.setCursor(QCursor(_ARROW_CUR)); b.setFixedSize(24,16)
        self._slider=QSlider(Qt.Orientation.Horizontal if _PYSIDE6 else Qt.Horizontal, self)
        self._slider.setMinimum(0); self._slider.setMaximum(0)
        self._slider.setSingleStep(_CELL_W); self._slider.setPageStep(_CELL_W*5); self._slider.setTracking(True)
        self._slider.setStyleSheet(
            'QSlider::groove:horizontal{height:5px;background:#111318;border:1px solid #1e2029;border-radius:3px;}'
            'QSlider::sub-page:horizontal{background:#4a4a4a;border-radius:3px;}'
            'QSlider::add-page:horizontal{background:#111318;border-radius:3px;}'
            'QSlider::handle:horizontal{width:14px;margin:-5px 0;border-radius:7px;background:#3a4050;border:1px solid #1e2029;}'
            'QSlider::handle:horizontal:hover{background:#525c70;}')
        btn_css=('QToolButton{color:#a8b0c4;background:#111318;border:1px solid #1e2029;border-radius:5px;font-size:12px;}'
                 'QToolButton:hover{background:#16181f;color:#c8d4e8;}'
                 'QToolButton:disabled{color:#3a4050;background:#0a0b0f;border-color:#111318;}')
        self._btn_first.setStyleSheet(btn_css); self._btn_last.setStyleSheet(btn_css)
        lay.addWidget(self._btn_first,0); lay.addWidget(self._slider,1); lay.addWidget(self._btn_last,0)
        self._btn_first.clicked.connect(self._go_first); self._btn_last.clicked.connect(self._go_last)
        self._slider.valueChanged.connect(self._on_slider)
        self._strip.sig_visible_changed.connect(self._sync); self._strip.sig_scroll_changed.connect(self._sync)
        self._sync(0)

    def _go_first(self):
        if not self._strip._visible: return
        self._strip.sig_navigate.emit(self._strip._visible[0]); self._strip.set_scroll_x(0)

    def _go_last(self):
        if not self._strip._visible: return
        self._strip.sig_navigate.emit(self._strip._visible[-1]); self._strip.set_scroll_x(self._strip.max_scroll())

    def _on_slider(self, v): self._strip.set_scroll_x(v)

    def _sync(self, _=0):
        mx=self._strip.max_scroll(); x=self._strip.scroll_x()
        self._slider.blockSignals(True)
        self._slider.setMaximum(mx); self._slider.setValue(max(0,min(x,mx)))
        self._slider.blockSignals(False); self._slider.setEnabled(mx>0)
        has=bool(self._strip._visible); self._btn_first.setEnabled(has); self._btn_last.setEnabled(has)
