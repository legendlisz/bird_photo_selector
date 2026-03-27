#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Li Qian
# All rights reserved.
"""
RAW Photo Viewer
================
版本历史请查看 CHANGELOG.md 文件。
"""

# ── 当前版本号 ────────────────────────────────────────────────────────
APP_VERSION = "3.70"

import sys, os, re, json, subprocess, queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

import numpy as np
import cv2

# ── 可选依赖 ──────────────────────────────────────────────────────────
try:
    from yolo_overlay import YoloOverlayManager, _get_model as _yolo_get_model, _YOLO_LOCK as _yolo_lock
except Exception:
    YoloOverlayManager = None
    _yolo_get_model    = None
    _yolo_lock         = None

try:
    from score_fusion import compute_fused_score
except Exception:
    compute_fused_score = None

try:
    from send2trash import send2trash as _send2trash
    HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

# ── Qt 兼容层 ─────────────────────────────────────────────────────────
from qt_compat import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QToolBar, QStatusBar,
    QMessageBox, QDialog, QCheckBox, QPushButton, QMenu,
    QToolButton,
    QPixmap, QColor, QFont, QIcon,
    QKeySequence, QAction, QPalette,
    QThread, QTimer, QSize,
    Signal, Qt,
    _PYSIDE6,
    _NO_FOCUS, _SP_EXPAND,
    _SOLID_LINE,
    _K_LEFT, _K_RIGHT, _K_DEL, _K_BACK,
    _MOD_CTRL, _MOD_SHIFT,
)

# ── 应用级常量 ────────────────────────────────────────────────────────
from app_constants import (
    SB_BG, SB_BORDER, SB_SEP,
    SB_LBL, SB_VAL, SB_VAL_ACTIVE,
    SB_VAL_WARN, SB_VAL_ERR, SB_VAL_AF_OK,
    SB_ET_OK, SB_VAL_PICK, SB_VAL_REJ,
    SB_FONT_SIZE,
    UI_FONT_FAMILY, UI_FONT_BASE_SIZE,
    SCORE_VERIFY_TOLERANCE,
)

# ── af_parser ─────────────────────────────────────────────────────────
from af_parser import (
    SUPPORTED_EXTS, BRAND_BY_EXT,
    set_exiftool_path,
)

# ── utils ─────────────────────────────────────────────────────────────
from utils import (
    FLAG_NONE, FLAG_PICK, FLAG_REJECT,
    _load_prefs, _save_prefs,
    find_exiftool,
    write_xmp_flag, _xmp_path, FlagCache, ScoreCache,
    delete_hint_text, move_to_trash_strict,
)

# ── 已提取模块 ────────────────────────────────────────────────────────
from image_view   import ImageView
from thumb_strip  import ThumbnailStrip, ThumbnailScrollBar
from workers      import (
    ImageData, LoadThread, ScanThread,
    CombinedBatchScoringThread, PreloadManager,
    retire_thread, cleanup_zombies_force,
)
from dialogs      import RejectPreviewDialog, SettingsDialog

# ── 首选项路径 & ExifTool ────────────────────────────────────────────
_PREFS_PATH   = Path.home() / '.raw_viewer_prefs.json'
EXIFTOOL_PATH = find_exiftool()
set_exiftool_path(EXIFTOOL_PATH)


def _resolve_app_icon() -> str:
    dirs = []
    if getattr(sys, 'frozen', False):
        dirs.append(os.path.dirname(sys.executable))
    if hasattr(sys, '_MEIPASS'):
        dirs.append(sys._MEIPASS)
    dirs.append(os.path.dirname(os.path.abspath(__file__)))
    seen = set()
    for d in dirs:
        if not d or d in seen:
            continue
        seen.add(d)
        p = os.path.join(d, 'bird_2.ico')
        if os.path.isfile(p):
            return p
    return ''


def _build_windows_appid() -> str:
    base = Path(sys.executable if getattr(sys, 'frozen', False) else __file__).stem
    safe = re.sub(r'[^A-Za-z0-9.]+', '.', base).strip('.')
    if not safe:
        safe = 'RAWViewer'
    return f'RAWViewer.{safe}'


def _sb(label, value, lc=SB_LBL, vc=SB_VAL):
    return (f"<span style='color:{lc};letter-spacing:0.3px'>{label}</span>"
            f"&nbsp;<span style='color:{vc};font-weight:500'>{value}</span>")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'RAW Photo Viewer  v{APP_VERSION}')
        self.resize(1280,840); self.setMinimumSize(860,600)

        self._files:         List[str]                    = []
        self._idx:           int                          = -1
        self._current_path:  Optional[str]                = None
        self._thread:        Optional[LoadThread]                    = None
        self._scan_thread:   Optional[ScanThread]                    = None
        self._score_thread:  Optional[CombinedBatchScoringThread]    = None
        self._scanning:      bool                              = False
        self._lock_zoom                                   = False
        self._prev_zoom_mode                              = 'fit'
        self._score_cache:   Optional[ScoreCache]         = None
        self._flag_cache:    Optional[FlagCache]          = None
        self._preloader:     PreloadManager               = PreloadManager(self)
        self._batch_resume_retries: int                   = 0
        self._batch_last_cached: int                      = -1

        self._wheel_accum: float = 0.0
        self._wheel_pending_steps: int = 0
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.timeout.connect(self._flush_wheel_nav)

        p=_load_prefs(_PREFS_PATH)
        self._confirm_delete: bool = p.get('confirm_delete', True)
        self._lock_zoom:      bool = p.get('lock_zoom', False)
        self._burst_fold:     bool = p.get('burst_fold', False)
        self._show_burst_groups: bool = p.get('show_burst_groups', True)
        self._burst_label_head_only: bool = p.get('burst_label_head_only', True)
        self._sort_by_burst_group: bool = p.get('sort_by_burst_group', False)
        _inner_mode = str(p.get('burst_group_inner_sort_mode', '') or '').strip().lower()
        if _inner_mode not in ('name', 'sharp', 'fused'):
            _inner_mode = 'sharp' if bool(p.get('burst_group_inner_sort_by_score', False)) else 'name'
        self._burst_group_inner_sort_mode: str = _inner_mode
        self._show_exif:      bool = p.get('show_exif', False)
        self._show_af:        bool = p.get('show_af', True)
        self._show_yolo_bird: bool = p.get('show_yolo_bird', True)
        self._show_yolo_eye:  bool = p.get('show_yolo_eye', True)
        self._yolo_bird_conf: float = float(p.get('yolo_bird_conf', 0.35))
        self._yolo_eye_conf:  float = float(p.get('yolo_eye_conf',  0.30))
        # 同步到 yolo_overlay 模块
        try:
            import yolo_overlay as _yo
            _yo.YOLO_BIRD_MIN_CONF = self._yolo_bird_conf
            _yo.YOLO_EYE_MIN_CONF  = self._yolo_eye_conf
        except ImportError:
            pass

        self._build_ui()
        self._build_menubar()
        self._build_toolbar()
        self._build_statusbar()
        self._refresh_nav_btns()

        self._menu_a_lock.blockSignals(True)
        self._menu_a_lock.setChecked(self._lock_zoom)
        self._menu_a_lock.blockSignals(False)
        if self._lock_zoom: self._refresh_lock_sb(True)

        self._a_burst_fold.blockSignals(True)
        self._a_burst_fold.setChecked(self._burst_fold)
        if hasattr(self.thumb_strip, 'set_burst_fold'):
            self.thumb_strip.set_burst_fold(self._burst_fold)
        self._a_burst_fold.blockSignals(False)

        self._menu_a_show_groups.blockSignals(True)
        self._menu_a_show_groups.setChecked(self._show_burst_groups)
        self._menu_a_show_groups.blockSignals(False)

        self._a_tb_show_groups.blockSignals(True)
        self._a_tb_show_groups.setChecked(self._show_burst_groups)
        self._a_tb_show_groups.blockSignals(False)

        if hasattr(self.thumb_strip, 'set_show_burst_groups'):
            self.thumb_strip.set_show_burst_groups(self._show_burst_groups)
        if hasattr(self.thumb_strip, 'set_burst_label_head_only'):
            self.thumb_strip.set_burst_label_head_only(self._burst_label_head_only)
        if hasattr(self.thumb_strip, 'set_sort_by_burst_group'):
            self.thumb_strip.set_sort_by_burst_group(self._sort_by_burst_group)
        if hasattr(self.thumb_strip, 'set_burst_group_inner_sort_mode'):
            self.thumb_strip.set_burst_group_inner_sort_mode(self._burst_group_inner_sort_mode)

        self._menu_a_group_label_head_only.blockSignals(True)
        self._menu_a_group_label_head_only.setChecked(self._burst_label_head_only)
        self._menu_a_group_label_head_only.blockSignals(False)


        self._menu_a_sort_by_group.blockSignals(True)
        self._menu_a_sort_by_group.setChecked(self._sort_by_burst_group)
        self._menu_a_sort_by_group.blockSignals(False)

        self._refresh_global_sort_btn()

        self._menu_a_inner_name.blockSignals(True)
        self._menu_a_inner_sharp.blockSignals(True)
        self._menu_a_inner_fused.blockSignals(True)
        self._menu_a_inner_name.setChecked(self._burst_group_inner_sort_mode == 'name')
        self._menu_a_inner_sharp.setChecked(self._burst_group_inner_sort_mode == 'sharp')
        self._menu_a_inner_fused.setChecked(self._burst_group_inner_sort_mode == 'fused')
        self._menu_a_inner_name.blockSignals(False)
        self._menu_a_inner_sharp.blockSignals(False)
        self._menu_a_inner_fused.blockSignals(False)
        self._refresh_group_inner_mode_btn()

        self.view.set_show_exif(self._show_exif)
        self.view.set_show_af(self._show_af)
        self.view.set_show_yolo_bird(self._show_yolo_bird)
        self.view.set_show_yolo_eye(self._show_yolo_eye)

        self._yolo_cache: Dict[str, dict] = {}
        self._yolo = YoloOverlayManager(self) if YoloOverlayManager else None
        if self._yolo:
            self._yolo.sig_ready.connect(self._on_yolo_ready)

        self._fused_cache: Dict[str, float] = {}
        self._current_qimage = None
        self._current_af_points = []
        self._current_ref_w = 0
        self._current_ref_h = 0
        self._current_live_sharpness = 0.0

        # 定期清理已结束的僵尸线程，防止长时间浏览后内存堆积
        self._zombie_timer = QTimer(self)
        self._zombie_timer.setInterval(5000)
        self._zombie_timer.timeout.connect(cleanup_zombies_force)
        self._zombie_timer.start()

    def _build_ui(self):
        c=QWidget(); self.setCentralWidget(c)
        lay=QVBoxLayout(c); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        self.view=ImageView(self)
        self.view.sig_prev.connect(self.go_prev)
        self.view.sig_next.connect(self.go_next)
        self.view.sig_wheel_steps.connect(self._on_wheel_steps)
        self.view.sig_zoom_chg.connect(self._update_sb)
        self.view.sig_delete_current.connect(self.delete_current_file)
        self.view.sig_reveal_current.connect(self.reveal_current_in_explorer)
        self.view.sig_exif_toggled.connect(self._on_exif_toggled_from_view)
        self.view.sig_af_toggled.connect(self._on_af_toggled_from_view)
        self.view.sig_yolo_bird_toggled.connect(self._on_yolo_bird_toggled_from_view)
        self.view.sig_yolo_eye_toggled.connect(self._on_yolo_eye_toggled_from_view)
        self.view.sig_rescore_selected.connect(self._rescore_selected_photos)
        self.view.sig_flag_pick.connect(lambda: self._toggle_flag(FLAG_PICK))
        self.view.sig_flag_reject.connect(lambda: self._toggle_flag(FLAG_REJECT))
        lay.addWidget(self.view,1)
        self.thumb_strip=ThumbnailStrip(self)
        self.thumb_strip.sig_navigate.connect(self._show)
        self.thumb_strip.sig_delete_paths.connect(self._delete_paths)
        self.thumb_strip.sig_visible_changed.connect(self._refresh_nav_btns)
        self.thumb_strip.sig_sort_selected.connect(self._menu_set_sort)
        self.thumb_strip.sig_group_inner_sort_selected.connect(self._on_group_inner_sort_mode_selected)
        self.thumb_strip.sig_expand_burst_group.connect(self._expand_burst_group_from_badge)

        thumb_box = QWidget(self)
        thumb_lay = QVBoxLayout(thumb_box)
        thumb_lay.setContentsMargins(0,0,0,0)
        thumb_lay.setSpacing(0)
        thumb_lay.addWidget(self.thumb_strip,0)
        self.thumb_scroll = ThumbnailScrollBar(self.thumb_strip, self)
        thumb_lay.addWidget(self.thumb_scroll,0)
        lay.addWidget(thumb_box,0)

    def _build_menubar(self):
        mb=self.menuBar(); mb.setNativeMenuBar(False)
        mb.setStyleSheet(f'''
            QMenuBar {{ background:{SB_BG}; color:#b7b7b7; font-size:13px;
                        padding:1px 0; border-bottom:1px solid {SB_BORDER}; }}
            QMenuBar::item {{ padding:3px 10px; border-radius:4px; }}
            QMenuBar::item:selected {{ background:#3a3a3a; }}
            QMenu {{ background:#2f2f2f; color:#bfbfbf; font-size:13px;
                     border:1px solid #3a3a3a; padding:3px 0; border-radius:6px; }}
            QMenu::item {{ padding:6px 22px 6px 14px; }}
            QMenu::item:selected {{ background:#444444; color:#f2f2f2; border-radius:3px; }}
            QMenu::separator {{ height:1px; background:#3a3a3a; margin:3px 10px; }}
            QMenu::item:disabled {{ color:#6f6f6f; }}
            QMenu::item:checked {{ color:#e2e2e2; }}
        ''')

        fm=mb.addMenu('文件(&F)')
        self._add_maction(fm,'打开文件(&O)','Ctrl+O',self.open_file)
        self._add_maction(fm,'打开文件夹(&D)','Ctrl+D',self.open_folder)
        fm.addSeparator()
        _dk_hint='⌫ Backspace' if sys.platform=='darwin' else 'Del'
        self._menu_a_delete=self._add_maction(fm,f'删除当前文件\t{_dk_hint}',None,self.delete_current_file)
        fm.addSeparator()
        self._add_maction(fm,'退出(&Q)','Ctrl+Q',self.close)

        em=mb.addMenu('编辑(&E)')
        self._add_maction(em,'上一张','Alt+Left',self.go_prev)
        self._add_maction(em,'下一张','Alt+Right',self.go_next)
        em.addSeparator()
        self._add_maction(em,'重新评分选中照片(&R)','Ctrl+Shift+R',self._rescore_selected_photos)
        em.addSeparator()
        self._add_maction(em,'设置(&S)…','Ctrl+,',self.open_settings)

        mm=mb.addMenu('标记(&M)')
        self._add_maction(mm,'★  标记选用  ( ~ )','`',lambda: self._toggle_flag(FLAG_PICK))
        self._add_maction(mm,'✕  标记待删除  ( d )','D',lambda: self._toggle_flag(FLAG_REJECT))
        self._add_maction(mm,'○  清除标记  ( u )','U',lambda: self._set_flag_current(FLAG_NONE))
        mm.addSeparator()
        self._add_maction(mm,'📋  查看所有选用照片','Ctrl+P',self._show_pick_list)
        self._add_maction(mm,'⚠  查看所有待删除照片','Ctrl+R',self._show_reject_list)
        mm.addSeparator()
        self._add_maction(mm,'🗑  删除全部"待删除"照片','Ctrl+Shift+D',self._delete_all_rejects)
        mm.addSeparator()
        flag_filter_menu = mm.addMenu('按标记过滤显示')
        self._flag_filter_actions: Dict[str, QAction] = {}
        for fkey, label in (('pick','★ 仅显示选用'), ('none','○ 显示未标记'), ('reject','✕ 显示待删除')):
            a=QAction(label, self); a.setCheckable(True); a.setChecked(True)
            a.triggered.connect(lambda checked, k=fkey: self._menu_toggle_flag_filter(k, checked))
            flag_filter_menu.addAction(a)
            self._flag_filter_actions[fkey]=a
        flag_filter_menu.addSeparator()
        flag_filter_menu.addAction('全部显示').triggered.connect(self._menu_show_all_flags)

        vm=mb.addMenu('视图(&V)')
        a_lock=QAction('锁定缩放(&L)',self); a_lock.setShortcut(QKeySequence('Ctrl+L'))
        a_lock.setCheckable(True); a_lock.setChecked(self._lock_zoom)
        a_lock.toggled.connect(self._on_lock_zoom_toggled); vm.addAction(a_lock)
        self._menu_a_lock=a_lock
        vm.addSeparator()
        a_exif=QAction('显示 EXIF 信息(&E)',self)
        a_exif.setShortcut(QKeySequence('Ctrl+E'))
        a_exif.setCheckable(True); a_exif.setChecked(self._show_exif)
        a_exif.triggered.connect(self._toggle_exif_overlay)
        vm.addAction(a_exif); self._menu_a_exif=a_exif

        a_show_groups=QAction('显示连拍分组线(&G)', self)
        a_show_groups.setCheckable(True)
        a_show_groups.setChecked(self._show_burst_groups)
        a_show_groups.triggered.connect(self._on_show_burst_groups_toggled)
        vm.addAction(a_show_groups); self._menu_a_show_groups=a_show_groups

        a_group_label_head_only=QAction('组号仅组首显示', self)
        a_group_label_head_only.setCheckable(True)
        a_group_label_head_only.setChecked(self._burst_label_head_only)
        a_group_label_head_only.triggered.connect(self._on_group_label_head_only_toggled)
        vm.addAction(a_group_label_head_only); self._menu_a_group_label_head_only=a_group_label_head_only

        a_sort_by_group=QAction('按连拍分组排序（组内最高分）', self)
        a_sort_by_group.setCheckable(True)
        a_sort_by_group.setChecked(self._sort_by_burst_group)
        a_sort_by_group.triggered.connect(self._on_sort_by_burst_group_toggled)
        vm.addAction(a_sort_by_group); self._menu_a_sort_by_group=a_sort_by_group

        inner_menu = vm.addMenu('🧩 组内排序')
        self._menu_a_inner_name=QAction('按文件名', self); self._menu_a_inner_name.setCheckable(True)
        self._menu_a_inner_sharp=QAction('按基础分', self); self._menu_a_inner_sharp.setCheckable(True)
        self._menu_a_inner_fused=QAction('按综合分', self); self._menu_a_inner_fused.setCheckable(True)
        self._menu_a_inner_name.triggered.connect(lambda checked: checked and self._on_group_inner_sort_mode_selected('name'))
        self._menu_a_inner_sharp.triggered.connect(lambda checked: checked and self._on_group_inner_sort_mode_selected('sharp'))
        self._menu_a_inner_fused.triggered.connect(lambda checked: checked and self._on_group_inner_sort_mode_selected('fused'))
        inner_menu.addAction(self._menu_a_inner_name)
        inner_menu.addAction(self._menu_a_inner_sharp)
        inner_menu.addAction(self._menu_a_inner_fused)

        vm.addSeparator()
        sort_menu=vm.addMenu('🌐 评分排序(&O)')
        self._sort_actions: Dict[str, QAction] = {}
        for key, label in (
            (None, '↕  文件名顺序'),
            ('desc','↓  基础分从高到低'),
            ('asc','↑  基础分从低到高'),
            ('fdesc','↓  综合得分从高到低'),
            ('fasc','↑  综合得分从低到高'),
        ):
            a=QAction(label, self); a.setCheckable(True); a.setChecked(key is None)
            a.triggered.connect(lambda checked, k=key: self._menu_set_sort(k))
            sort_menu.addAction(a); self._sort_actions[str(key)]=a
        filter_menu=vm.addMenu('评分过滤(&F)')
        mode_menu = filter_menu.addMenu('过滤模式')
        self._filter_mode_actions: Dict[str, QAction] = {}
        for mode, label in (
            ('off', '取消过滤'),
            ('sharp_only', '过滤基础分'),
            ('fused_only', '过滤综合分'),
            ('both', '双过滤'),
        ):
            a=QAction(label, self); a.setCheckable(True); a.setChecked(self.thumb_strip._filter_mode == mode)
            a.triggered.connect(lambda checked, m=mode: checked and self._menu_set_filter_mode(m))
            mode_menu.addAction(a); self._filter_mode_actions[mode]=a
        filter_menu.addSeparator()
        tier_menu = filter_menu.addMenu('评分档位')
        self._filter_actions: Dict[str, QAction] = {}
        for tier, label in (('green','🟢 绿色'),('yellow','🟡 黄色'),('red','🔴 红色')):
            a=QAction(label, self); a.setCheckable(True); a.setChecked(True)
            a.triggered.connect(lambda checked, t=tier: self._menu_toggle_filter(t, checked))
            tier_menu.addAction(a); self._filter_actions[tier]=a
        tier_menu.addSeparator()
        tier_menu.addAction('全部颜色').triggered.connect(self._menu_show_all_tiers)


        hm = mb.addMenu('帮助(&H)')
        self._add_maction(hm, f'关于 RAW Photo Viewer…', None, self._show_about)

    def _add_maction(self,menu,text,sc,fn):
        a=QAction(text,self)
        if sc: a.setShortcut(QKeySequence(sc))
        a.triggered.connect(fn); menu.addAction(a); return a

    def _build_toolbar(self):
        tb=QToolBar('主工具栏',self); tb.setMovable(False); tb.setFloatable(False)
        tb.setIconSize(QSize(20,20))
        tb.setStyleSheet(f'''
            QToolBar {{ background:{SB_BG}; spacing:4px; padding:4px 8px;
                        border-bottom:1px solid {SB_BORDER}; }}
            QToolButton {{ font-size:13px; padding:4px 11px; border-radius:5px; color:#b6b6b6; border:1px solid transparent; }}
            QToolButton:hover   {{ background:#3a3a3a; color:#d9d9d9; border:1px solid #4a4a4a; }}
            QToolButton:checked {{ background:transparent; color:#d2d2d2; border:1px solid transparent; font-weight:600; }}
            QToolButton:checked:hover {{ background:transparent; color:#dddddd; border:1px solid transparent; }}
            QToolButton:disabled{{ color:#363d4a; border:1px solid transparent; }}
        ''')
        self.addToolBar(tb)
        def mk(txt,sc,fn,tip=''):
            a=QAction(txt,self)
            if sc: a.setShortcut(QKeySequence(sc))
            if tip: a.setToolTip(tip)
            a.triggered.connect(fn); tb.addAction(a); return a
        mk('□ 文件夹','Ctrl+D',self.open_folder)
        tb.addSeparator()

        self._a_prev=QAction('◀ 上一张', self)
        self._a_prev.setShortcut(QKeySequence('Alt+Left'))
        self._a_prev.triggered.connect(self.go_prev)
        self.addAction(self._a_prev)

        self._a_next=QAction('▶ 下一张', self)
        self._a_next.setShortcut(QKeySequence('Alt+Right'))
        self._a_next.triggered.connect(self.go_next)
        self.addAction(self._a_next)

        self._a_tb_pick=QAction('★ 选用',self)
        self._a_tb_pick.setToolTip('标记选用 ( ~ )')
        self._a_tb_pick.setCheckable(True)
        self._a_tb_pick.triggered.connect(lambda: self._toggle_flag(FLAG_PICK))

        self._a_tb_reject=QAction('✕ 待删',self)
        self._a_tb_reject.setToolTip('标记待删除 ( d )')
        self._a_tb_reject.setCheckable(True)
        self._a_tb_reject.triggered.connect(lambda: self._toggle_flag(FLAG_REJECT))

        self._a_tb_show_groups=QAction('▦ 连拍分组',self)
        self._a_tb_show_groups.setCheckable(True)
        self._a_tb_show_groups.setToolTip('显示/隐藏连拍分组')
        self._a_tb_show_groups.toggled.connect(self._on_show_burst_groups_toggled)
        tb.addAction(self._a_tb_show_groups)

        tb.addSeparator()
        self._a_burst_fold=QAction('▤ 连拍堆叠',self)
        self._a_burst_fold.setShortcut(QKeySequence('B'))
        self._a_burst_fold.setCheckable(True)
        self._a_burst_fold.setToolTip('将相近时间的连拍照片折叠，只显示评分最高的一张 ( B / - 折叠 / = 展开 )')
        self._a_burst_fold.toggled.connect(self._on_burst_fold_toggled)
        tb.addAction(self._a_burst_fold)

        self._a_burst_fold_on=QAction(self)
        self._a_burst_fold_on.setShortcut(QKeySequence('-'))
        self._a_burst_fold_on.triggered.connect(lambda: self._a_burst_fold.setChecked(True))
        self.addAction(self._a_burst_fold_on)

        self._a_burst_fold_off=QAction(self)
        self._a_burst_fold_off.setShortcut(QKeySequence('='))
        self._a_burst_fold_off.triggered.connect(lambda: self._a_burst_fold.setChecked(False))
        self.addAction(self._a_burst_fold_off)

        tb.addSeparator()

        self._tb_global_sort_btn=QToolButton(self)
        self._tb_global_sort_btn.setText('⇅ 排序:名')
        self._tb_global_sort_btn.setToolTip('全局排序：下拉选择文件名/基础分/综合分')
        self._tb_global_sort_btn.setPopupMode(QToolButton.InstantPopup)
        self._tb_sort_menu=QMenu(self._tb_global_sort_btn)
        self._tb_sort_actions: Dict[str, QAction] = {}
        for key, label in (
            (None, '↕ 文件名顺序'),
            ('desc', '↓ 基础分从高到低'),
            ('fdesc', '↓ 综合得分从高到低'),
        ):
            a=QAction(label, self); a.setCheckable(True); a.setChecked(key is None)
            a.triggered.connect(lambda checked, k=key: self._menu_set_sort(k))
            self._tb_sort_menu.addAction(a); self._tb_sort_actions[str(key)]=a
        self._tb_global_sort_btn.setMenu(self._tb_sort_menu)
        tb.addWidget(self._tb_global_sort_btn)

        self._tb_group_inner_sort_btn=QToolButton(self)
        self._tb_group_inner_sort_btn.setText('≣ 组内:名')
        self._tb_group_inner_sort_btn.setToolTip('组内排序：下拉选择文件名/基础分/综合分')
        self._tb_group_inner_sort_btn.setPopupMode(QToolButton.InstantPopup)
        self._tb_group_inner_sort_menu=QMenu(self._tb_group_inner_sort_btn)
        self._tb_group_inner_sort_actions: Dict[str, QAction] = {}
        for key, label in (
            ('name', '≣ 文件名顺序'),
            ('sharp', '≣ 基础分从高到低'),
            ('fused', '≣ 综合得分从高到低'),
        ):
            a=QAction(label, self); a.setCheckable(True); a.setChecked(key == 'name')
            a.triggered.connect(lambda checked, k=key: checked and self._on_group_inner_sort_mode_selected(k))
            self._tb_group_inner_sort_menu.addAction(a); self._tb_group_inner_sort_actions[key]=a
        self._tb_group_inner_sort_btn.setMenu(self._tb_group_inner_sort_menu)
        tb.addWidget(self._tb_group_inner_sort_btn)

        tb.addSeparator()
        self._a_tb_yolo_eye=QAction('鸟眼',self)
        self._a_tb_yolo_eye.setCheckable(True); self._a_tb_yolo_eye.setChecked(self._show_yolo_eye)
        self._a_tb_yolo_eye.setToolTip('显示/隐藏 鸟眼范围')
        self._a_tb_yolo_eye.toggled.connect(self._toggle_yolo_eye_overlay)
        tb.addAction(self._a_tb_yolo_eye)

        self._a_tb_yolo_bird=QAction('鸟身',self)
        self._a_tb_yolo_bird.setCheckable(True); self._a_tb_yolo_bird.setChecked(self._show_yolo_bird)
        self._a_tb_yolo_bird.setToolTip('显示/隐藏 鸟身轮廓')
        self._a_tb_yolo_bird.toggled.connect(self._toggle_yolo_bird_overlay)
        tb.addAction(self._a_tb_yolo_bird)

        self._a_tb_af=QAction('对焦',self)
        self._a_tb_af.setCheckable(True); self._a_tb_af.setChecked(self._show_af)
        self._a_tb_af.setToolTip('显示/隐藏 对焦范围')
        self._a_tb_af.toggled.connect(self._toggle_af_overlay)
        tb.addAction(self._a_tb_af)

        self._a_tb_exif=QAction('EXIF',self)
        self._a_tb_exif.setShortcut(QKeySequence('Ctrl+E'))
        self._a_tb_exif.setCheckable(True); self._a_tb_exif.setChecked(self._show_exif)
        self._a_tb_exif.setToolTip('显示/隐藏 EXIF 信息覆盖层 (Ctrl+E)')
        self._a_tb_exif.toggled.connect(self._toggle_exif_overlay)
        tb.addAction(self._a_tb_exif)
        tb.addSeparator()
        _dk='Backspace' if sys.platform=='darwin' else 'Delete'
        self._a_delete=QAction('删除', self)
        self._a_delete.setShortcut(QKeySequence(_dk))
        self._a_delete.triggered.connect(self.delete_current_file)
        self.addAction(self._a_delete)
        self._a_delete.setEnabled(False)

        self._a_confirm_del=QAction('询删:开' if self._confirm_delete else '询删:关',self)
        self._a_confirm_del.setCheckable(True); self._a_confirm_del.setChecked(self._confirm_delete)
        self._a_confirm_del.setToolTip('删除前询问 开/关')
        self._a_confirm_del.toggled.connect(self._on_confirm_del_toggled)
        tb.addSeparator()
        self._lbl_fname=QLabel('  未打开文件  ')
        self._lbl_fname.setStyleSheet('color:#9a9a9a;padding:0 8px;font-size:11px;')
        tb.addWidget(self._lbl_fname)

    def _build_statusbar(self):
        sb=self.statusBar(); sb.setFixedHeight(24); sb.setFocusPolicy(_NO_FOCUS)
        sb.setStyleSheet(f'QStatusBar{{background:{SB_BG};border-top:1px solid {SB_BORDER};}}'
                         'QStatusBar::item{border:none;}')
        FS=SB_FONT_SIZE; sty=f'font-size:{FS}px;padding:0 4px;background:transparent;'
        def ml(html=''):
            l=QLabel(); l.setStyleSheet(sty); l.setFocusPolicy(_NO_FOCUS)
            if html: l.setText(html)
            return l
        c=QWidget(); h=QHBoxLayout(c); h.setContentsMargins(8,0,8,0); h.setSpacing(0)
        c.setFocusPolicy(_NO_FOCUS)
        self._sb_idx   =ml(_sb('序号','—'))
        self._sb_size  =ml(); self._sb_zoom=ml(); self._sb_scan=ml(); self._sb_batch=ml(); self._sb_fused=ml()
        self._sb_flag  =ml()
        for w in (self._sb_idx,_msep(),self._sb_size,_msep(),self._sb_zoom,
                  _msep(),self._sb_scan,_msep(),self._sb_batch,_msep(),self._sb_fused,_msep(),self._sb_flag):
            h.addWidget(w)
        h.addStretch(1)
        self._sb_lockzoom=ml(); h.addWidget(self._sb_lockzoom); h.addWidget(_msep())
        et=(ml(_sb('ExifTool','已就绪',vc=SB_VAL)) if EXIFTOOL_PATH
            else ml(_sb('ExifTool','未安装',vc=SB_VAL_ERR)))
        if EXIFTOOL_PATH: et.setToolTip(f'路径：{EXIFTOOL_PATH}')
        h.addWidget(et)
        h.addWidget(_msep())
        # YOLO 状态
        if YoloOverlayManager is None:
            yolo_lbl = ml(_sb('YOLO','未安装',vc=SB_VAL_ERR))
            yolo_lbl.setToolTip('未找到 ultralytics，请运行：pip install ultralytics')
        else:
            from pathlib import Path as _P
            _base = _P(__file__).resolve().parent
            _bm = _base / 'models' / 'yolov8x-seg.pt'
            _em = _base / 'models' / 'best.pt'
            if _bm.is_file() and _em.is_file():
                yolo_lbl = ml(_sb('YOLO','已就绪',vc=SB_VAL_AF_OK))
                yolo_lbl.setToolTip(f'鸟身模型：{_bm}\n鸟眼模型：{_em}')
            else:
                missing = []
                if not _bm.is_file(): missing.append(f'缺少 {_bm.name}')
                if not _em.is_file(): missing.append(f'缺少 {_em.name}')
                yolo_lbl = ml(_sb('YOLO','模型缺失',vc=SB_VAL_ERR))
                yolo_lbl.setToolTip('models/ 目录下缺少模型文件：\n' + '\n'.join(missing))
        self._sb_yolo = yolo_lbl
        h.addWidget(yolo_lbl)
        sb.addWidget(c,1)

    # ════════════════════════════════════════════════════════════════
    #  § 标记核心逻辑
    # ════════════════════════════════════════════════════════════════

    def _selected_file_indices(self) -> List[int]:
        if not self._files: return []
        sel=set()
        ts=getattr(self,'thumb_strip',None)
        if ts is not None and hasattr(ts,'_selected'):
            try: sel=set(ts._selected)
            except Exception: sel=set()
        idxs=[i for i in sorted(sel) if 0<=i<len(self._files)]
        if not idxs and self._idx>=0 and self._idx<len(self._files):
            idxs=[self._idx]
        return idxs

    def _rescore_selected_photos(self):
        if not self._files or self._score_cache is None:
            return
        idxs = self._selected_file_indices()
        if not idxs:
            return
        paths = [self._files[i] for i in idxs if 0 <= i < len(self._files)]
        if not paths:
            return
        for p in paths:
            self._yolo_cache.pop(p, None)
        retire_thread(self._score_thread); self._score_thread = None
        self._sb_batch.setText(_sb('评分', f'重评 {len(paths)} 张', vc=SB_VAL_WARN))
        self._sb_fused.setText(_sb('综合', '重评中', vc=SB_VAL_WARN))
        self.statusBar().showMessage(f'开始重评 {len(paths)} 张：鸟身/鸟眼识别 + 综合评分', 3000)
        t = CombinedBatchScoringThread(paths, self._score_cache, force_sharp=True, force_fused=True)
        t.sig_progress.connect(self._on_combined_progress)
        t.sig_done.connect(self._on_combined_done)
        self._score_thread = t
        t.start()

    def _apply_flags(self, idx_to_flag: Dict[int,int]):
        if not self._files or not idx_to_flag: return
        for idx, flag in idx_to_flag.items():
            if idx < 0 or idx >= len(self._files):
                continue
            path=self._files[idx]
            if self._flag_cache:
                self._flag_cache.set(path, flag)
            write_xmp_flag(path, flag)
        if self._flag_cache:
            self._flag_cache.save_if_dirty()
        if hasattr(self.thumb_strip, 'set_flags_bulk'):
            self.thumb_strip.set_flags_bulk(idx_to_flag)
        else:
            for idx, flag in idx_to_flag.items():
                self.thumb_strip.set_flag(idx, flag)
        if self._idx in idx_to_flag:
            cur_flag=idx_to_flag[self._idx]
            self.view.set_flag(cur_flag)
            self._refresh_flag_sb(cur_flag)
            self._refresh_flag_toolbar(cur_flag)

    def _toggle_flag(self, flag: int):
        if not self._files: return
        idxs=self._selected_file_indices()
        if not idxs: return
        idx_to_flag: Dict[int,int] = {}
        for idx in idxs:
            path=self._files[idx]
            cur=self._flag_cache.get(path) if self._flag_cache else FLAG_NONE
            idx_to_flag[idx] = (FLAG_NONE if cur == flag else flag)
        self._apply_flags(idx_to_flag)
        if len(idxs)==1:
            self.go_next()
        else:
            tag='★ 选用' if flag==FLAG_PICK else ('✕ 待删除' if flag==FLAG_REJECT else '标记')
            self.statusBar().showMessage(f'已批量切换 {len(idxs)} 张：{tag}', 2500)

    def _set_flag_current(self, flag: int):
        if not self._files: return
        idxs=self._selected_file_indices()
        if not idxs: return
        idx_to_flag={idx: flag for idx in idxs}
        self._apply_flags(idx_to_flag)
        if len(idxs)==1:
            self.go_next()
        else:
            if flag==FLAG_NONE:
                self.statusBar().showMessage(f'已清除 {len(idxs)} 张标记', 2500)
            elif flag==FLAG_PICK:
                self.statusBar().showMessage(f'已标记 {len(idxs)} 张为 ★ 选用', 2500)
            elif flag==FLAG_REJECT:
                self.statusBar().showMessage(f'已标记 {len(idxs)} 张为 ✕ 待删除', 2500)

    def _refresh_flag_sb(self, flag: int):
        FS = SB_FONT_SIZE
        if flag == FLAG_PICK:
            self._sb_flag.setText(_sb('标记','★ 选用', vc=SB_VAL_PICK))
        elif flag == FLAG_REJECT:
            self._sb_flag.setText(_sb('标记','✕ 待删除', vc=SB_VAL_REJ))
        else:
            self._sb_flag.setText('')
        self._sb_flag.setStyleSheet(f'font-size:{FS}px;padding:0 6px;background:transparent;')

    def _refresh_flag_toolbar(self, flag: int):
        for a in (self._a_tb_pick, self._a_tb_reject):
            a.blockSignals(True)
        self._a_tb_pick  .setChecked(flag == FLAG_PICK)
        self._a_tb_reject.setChecked(flag == FLAG_REJECT)
        for a in (self._a_tb_pick, self._a_tb_reject):
            a.blockSignals(False)

    def _show_pick_list(self):
        if not self._flag_cache or not self._files:
            QMessageBox.information(self,'选用照片','尚未打开目录或没有标记选用的照片。'); return
        picks = self._flag_cache.all_with_flag(self._files, FLAG_PICK)
        if not picks:
            QMessageBox.information(self,'选用照片','当前目录没有标记为选用的照片。'); return
        names = '\n'.join(f'  ★  {os.path.basename(p)}' for p in picks)
        QMessageBox.information(self,
            f'选用照片（共 {len(picks)} 张）',
            f'以下照片已标记为选用（已写入 XMP）：\n\n{names}\n\n'
            f'Lightroom 中：图库模块 → 元数据 → 从文件读取，即可同步标记。')

    def _show_reject_list(self):
        if not self._flag_cache or not self._files:
            QMessageBox.information(self,'待删除照片','尚未打开目录或没有标记待删除的照片。'); return
        rejects = self._flag_cache.all_with_flag(self._files, FLAG_REJECT)
        if not rejects:
            QMessageBox.information(self,'待删除照片','当前目录没有标记为待删除的照片。'); return
        dlg = RejectPreviewDialog(rejects, self._flag_cache,
                                  parent=self, score_cache=self._score_cache)
        dlg.sig_unmark.connect(self._on_reject_dlg_unmark)
        dlg.sig_delete.connect(self._on_reject_dlg_delete)
        dlg.exec() if _PYSIDE6 else dlg.exec_()

    def _on_reject_dlg_unmark(self, path: str):
        if self._flag_cache:
            self._flag_cache.set(path, FLAG_NONE)
            self._flag_cache.save()
        write_xmp_flag(path, FLAG_NONE)
        if self._current_path and os.path.basename(self._current_path) == os.path.basename(path):
            self.view.set_flag(FLAG_NONE)
            self._refresh_flag_sb(FLAG_NONE)
            self._refresh_flag_toolbar(FLAG_NONE)
        for i, f in enumerate(self._files):
            if f == path:
                self.thumb_strip.set_flag(i, FLAG_NONE)
                break

    def _on_reject_dlg_delete(self, paths: list):
        self._delete_paths(paths)

    def _delete_all_rejects(self):
        if not self._flag_cache or not self._files: return
        rejects = self._flag_cache.all_with_flag(self._files, FLAG_REJECT)
        if not rejects:
            self.statusBar().showMessage('没有待删除照片', 3000); return
        n = len(rejects)
        msg = QMessageBox(self); msg.setWindowTitle('删除全部待删除照片')
        tn = delete_hint_text(HAS_SEND2TRASH)
        msg.setText(f'确定要删除全部 {n} 张标记为"待删除"的照片？\n\n{tn}')
        if _PYSIDE6:
            msg.setIcon(QMessageBox.Icon.Warning)
            _Yes=QMessageBox.StandardButton.Yes; _No=QMessageBox.StandardButton.No
        else:
            msg.setIcon(QMessageBox.Warning); _Yes=QMessageBox.Yes; _No=QMessageBox.No
        msg.setStandardButtons(_Yes|_No); msg.setDefaultButton(_No)
        if (msg.exec() if _PYSIDE6 else msg.exec_()) == _Yes:
            self._delete_paths(rejects)

    def _menu_toggle_flag_filter(self, fkey: str, checked: bool):
        ts = self.thumb_strip
        if checked: ts._filter_flags.add(fkey)
        else:       ts._filter_flags.discard(fkey)
        ts._rebuild_visible(); ts._scroll_to(ts._current); ts.update()

    def _menu_show_all_flags(self):
        ts = self.thumb_strip
        ts._filter_flags = {'pick','none','reject'}
        for a in self._flag_filter_actions.values():
            a.blockSignals(True); a.setChecked(True); a.blockSignals(False)
        ts._rebuild_visible(); ts._scroll_to(ts._current); ts.update()

    # ════════════════════════════════════════════════════════════════
    #  § 其余方法
    # ════════════════════════════════════════════════════════════════

    def open_settings(self):
        dlg=SettingsDialog(self)
        dlg._btn_rescore.clicked.connect(lambda:(self._start_combined_scoring(force_sharp=True, force_fused=True),dlg.reject()))
        ok=(dlg.exec()==QDialog.DialogCode.Accepted if _PYSIDE6 else dlg.exec()==QDialog.Accepted)
        if not ok: return
        changed=[]
        if dlg.confirm_delete!=self._confirm_delete:
            self._confirm_delete=dlg.confirm_delete
            self._a_confirm_del.blockSignals(True); self._a_confirm_del.setChecked(self._confirm_delete)
            self._a_confirm_del.setText('询删:开' if self._confirm_delete else '询删:关')
            self._a_confirm_del.blockSignals(False); changed.append('删除确认')
        if dlg.lock_zoom!=self._lock_zoom:
            self._set_lock_zoom(dlg.lock_zoom); changed.append('锁定缩放')
        if dlg.show_exif!=self._show_exif:
            self._toggle_exif_overlay(dlg.show_exif); changed.append('EXIF 覆盖层')
        if dlg.burst_label_head_only!=self._burst_label_head_only:
            self._on_group_label_head_only_toggled(dlg.burst_label_head_only); changed.append('组号显示')
        new_bird_conf = round(dlg.yolo_bird_conf, 2)
        new_eye_conf  = round(dlg.yolo_eye_conf,  2)
        if new_bird_conf != self._yolo_bird_conf or new_eye_conf != self._yolo_eye_conf:
            self._yolo_bird_conf = new_bird_conf
            self._yolo_eye_conf  = new_eye_conf
            try:
                import yolo_overlay as _yo
                _yo.YOLO_BIRD_MIN_CONF = self._yolo_bird_conf
                _yo.YOLO_EYE_MIN_CONF  = self._yolo_eye_conf
            except ImportError:
                pass
            changed.append('YOLO 阈值')
        p=_load_prefs(_PREFS_PATH)
        p['confirm_delete']=self._confirm_delete; p['lock_zoom']=self._lock_zoom
        p['show_exif']=self._show_exif; p['burst_label_head_only']=self._burst_label_head_only
        p['yolo_bird_conf']=self._yolo_bird_conf; p['yolo_eye_conf']=self._yolo_eye_conf
        _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('设置已保存'+(f'：{" / ".join(changed)}' if changed else '，无更改'),3000)

    def _show_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('关于')
        if _PYSIDE6:
            msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(
            f'<b style="font-size:15px;">RAW Photo Viewer</b>'
            f'<br><span style="color:#525c70;">版本 {APP_VERSION}</span>'
            f'<br><br>'
            f'<span style="color:#525c70;">Copyright © 2026 Li Qian</span>'
        )
        if _PYSIDE6:
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        else:
            msg.setStandardButtons(QMessageBox.Ok)
        msg.exec() if _PYSIDE6 else msg.exec_()

    def _set_lock_zoom(self,checked):
        self._lock_zoom=checked
        if hasattr(self, '_menu_a_lock'):
            self._menu_a_lock.blockSignals(True)
            self._menu_a_lock.setChecked(checked)
            self._menu_a_lock.blockSignals(False)
        self._refresh_lock_sb(checked)
        if checked and self.view.zoom_mode=='100': self.view.enter_100_at_af()

    def _refresh_lock_sb(self,on):
        FS=SB_FONT_SIZE
        self._sb_lockzoom.setText(_sb('缩放','已锁定',vc=SB_VAL) if on else '')
        self._sb_lockzoom.setStyleSheet(f'font-size:{FS}px;padding:0 6px;')

    def _on_lock_zoom_toggled(self,checked):
        self._set_lock_zoom(checked)
        p=_load_prefs(_PREFS_PATH); p['lock_zoom']=checked; _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('锁定缩放已启用' if checked else '锁定缩放已关闭',3000)

    def _expand_burst_group_from_badge(self, idx: int):
        if idx < 0:
            return
        if hasattr(self, '_a_burst_fold') and self._a_burst_fold.isChecked():
            self._a_burst_fold.setChecked(False)
        else:
            self._burst_fold = False
            if hasattr(self.thumb_strip, 'set_burst_fold'):
                self.thumb_strip.set_burst_fold(False)
        if 0 <= idx < len(self._files):
            if self._idx != idx:
                self._show(idx)
            else:
                self.thumb_strip.set_current(idx)
        self.statusBar().showMessage('已展开当前连拍组', 1800)

    def _on_burst_fold_toggled(self, checked: bool):
        self._burst_fold = checked

        focus_idx = self._idx if (self._idx is not None and self._idx >= 0) else -1
        if hasattr(self.thumb_strip, 'get_burst_best_for') and focus_idx >= 0:
            focus_idx = self.thumb_strip.get_burst_best_for(focus_idx)

        if hasattr(self.thumb_strip, 'set_burst_fold'):
            self.thumb_strip.set_burst_fold(checked)

        if focus_idx >= 0 and focus_idx < len(self._files):
            if self._idx != focus_idx:
                self._show(focus_idx)
            else:
                self.thumb_strip.set_current(focus_idx)

        p = _load_prefs(_PREFS_PATH)
        p['burst_fold'] = checked
        _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('连拍折叠已' + ('开启' if checked else '关闭'), 2500)

    def _on_show_burst_groups_toggled(self, checked: bool):
        self._show_burst_groups = bool(checked)
        if hasattr(self, '_menu_a_show_groups'):
            self._menu_a_show_groups.blockSignals(True)
            self._menu_a_show_groups.setChecked(self._show_burst_groups)
            self._menu_a_show_groups.blockSignals(False)
        if hasattr(self, '_a_tb_show_groups'):
            self._a_tb_show_groups.blockSignals(True)
            self._a_tb_show_groups.setChecked(self._show_burst_groups)
            self._a_tb_show_groups.blockSignals(False)
        if hasattr(self.thumb_strip, 'set_show_burst_groups'):
            self.thumb_strip.set_show_burst_groups(self._show_burst_groups)
        p = _load_prefs(_PREFS_PATH)
        p['show_burst_groups'] = self._show_burst_groups
        _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('连拍分组线已' + ('显示' if checked else '隐藏'), 2000)

    def _on_group_label_head_only_toggled(self, checked: bool):
        self._burst_label_head_only = bool(checked)
        if hasattr(self, '_menu_a_group_label_head_only'):
            self._menu_a_group_label_head_only.blockSignals(True)
            self._menu_a_group_label_head_only.setChecked(self._burst_label_head_only)
            self._menu_a_group_label_head_only.blockSignals(False)
        if hasattr(self, '_a_tb_group_label_head_only'):
            self._a_tb_group_label_head_only.blockSignals(True)
            self._a_tb_group_label_head_only.setChecked(self._burst_label_head_only)
            self._a_tb_group_label_head_only.blockSignals(False)
        if hasattr(self.thumb_strip, 'set_burst_label_head_only'):
            self.thumb_strip.set_burst_label_head_only(self._burst_label_head_only)
        p = _load_prefs(_PREFS_PATH)
        p['burst_label_head_only'] = self._burst_label_head_only
        _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('组号显示模式：' + ('仅组首' if checked else '全部显示'), 2000)

    def _on_sort_by_burst_group_toggled(self, checked: bool):
        self._sort_by_burst_group = bool(checked)
        if hasattr(self, '_menu_a_sort_by_group'):
            self._menu_a_sort_by_group.blockSignals(True)
            self._menu_a_sort_by_group.setChecked(self._sort_by_burst_group)
            self._menu_a_sort_by_group.blockSignals(False)
        if hasattr(self, '_a_tb_sort_by_group'):
            self._a_tb_sort_by_group.blockSignals(True)
            self._a_tb_sort_by_group.setChecked(self._sort_by_burst_group)
            self._a_tb_sort_by_group.blockSignals(False)
        if hasattr(self.thumb_strip, 'set_sort_by_burst_group'):
            self.thumb_strip.set_sort_by_burst_group(self._sort_by_burst_group)
        p = _load_prefs(_PREFS_PATH)
        p['sort_by_burst_group'] = self._sort_by_burst_group
        _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('分组排序已' + ('开启' if checked else '关闭'), 2000)

    def _refresh_group_inner_mode_btn(self):
        m = self._burst_group_inner_sort_mode
        txt = '≣ 组内:名' if m == 'name' else ('≣ 组内:清' if m == 'sharp' else '≣ 组内:综')
        if hasattr(self, '_tb_group_inner_sort_btn'):
            self._tb_group_inner_sort_btn.setText(txt)
        if hasattr(self, '_tb_group_inner_sort_actions'):
            for k, a in self._tb_group_inner_sort_actions.items():
                a.blockSignals(True); a.setChecked(m == k); a.blockSignals(False)
        if hasattr(self, '_a_tb_group_inner_mode'):
            self._a_tb_group_inner_mode.setText(txt)

    def _set_group_inner_sort_mode(self, mode: str, persist: bool = True):
        mode = (mode or 'name').strip().lower()
        if mode not in ('name', 'sharp', 'fused'):
            mode = 'name'
        self._burst_group_inner_sort_mode = mode
        if hasattr(self.thumb_strip, 'set_burst_group_inner_sort_mode'):
            self.thumb_strip.set_burst_group_inner_sort_mode(mode)
        if hasattr(self, '_menu_a_inner_name'):
            self._menu_a_inner_name.blockSignals(True); self._menu_a_inner_name.setChecked(mode == 'name'); self._menu_a_inner_name.blockSignals(False)
            self._menu_a_inner_sharp.blockSignals(True); self._menu_a_inner_sharp.setChecked(mode == 'sharp'); self._menu_a_inner_sharp.blockSignals(False)
            self._menu_a_inner_fused.blockSignals(True); self._menu_a_inner_fused.setChecked(mode == 'fused'); self._menu_a_inner_fused.blockSignals(False)
        self._refresh_group_inner_mode_btn()
        if persist:
            p = _load_prefs(_PREFS_PATH)
            p['burst_group_inner_sort_mode'] = mode
            _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('组内排序：' + ({'name':'文件名','sharp':'基础分','fused':'综合分'}.get(mode, '文件名')), 2000)

    def _on_group_inner_sort_mode_selected(self, mode: str):
        self._set_group_inner_sort_mode(mode, persist=True)

    def _on_confirm_del_toggled(self,checked):
        self._confirm_delete=checked; self._a_confirm_del.setText('询删:开' if checked else '询删:关')
        p=_load_prefs(_PREFS_PATH); p['confirm_delete']=checked; _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('删除前确认已'+('开启' if checked else '关闭'),2500)

    def _toggle_exif_overlay(self, checked: bool):
        self._show_exif = checked; self.view.set_show_exif(checked)
        for a in (self._menu_a_exif, self._a_tb_exif):
            a.blockSignals(True); a.setChecked(checked); a.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_exif']=checked; _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('EXIF 信息已'+('显示' if checked else '隐藏'),2000)

    def _on_exif_toggled_from_view(self, checked: bool):
        self._show_exif = checked
        for a in (self._menu_a_exif, self._a_tb_exif):
            a.blockSignals(True); a.setChecked(checked); a.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_exif']=checked; _save_prefs(_PREFS_PATH, p)

    def _toggle_af_overlay(self, checked: bool):
        self._show_af = checked; self.view.set_show_af(checked)
        if hasattr(self, '_a_tb_af'):
            self._a_tb_af.blockSignals(True); self._a_tb_af.setChecked(checked); self._a_tb_af.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_af']=checked; _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('对焦范围已'+('显示' if checked else '隐藏'),2000)

    def _toggle_yolo_bird_overlay(self, checked: bool):
        self._show_yolo_bird = checked; self.view.set_show_yolo_bird(checked)
        if hasattr(self, '_a_tb_yolo_bird'):
            self._a_tb_yolo_bird.blockSignals(True); self._a_tb_yolo_bird.setChecked(checked); self._a_tb_yolo_bird.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_yolo_bird']=checked; _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('鸟身轮廓已'+('显示' if checked else '隐藏'),2000)

    def _toggle_yolo_eye_overlay(self, checked: bool):
        self._show_yolo_eye = checked; self.view.set_show_yolo_eye(checked)
        if hasattr(self, '_a_tb_yolo_eye'):
            self._a_tb_yolo_eye.blockSignals(True); self._a_tb_yolo_eye.setChecked(checked); self._a_tb_yolo_eye.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_yolo_eye']=checked; _save_prefs(_PREFS_PATH, p)
        self.statusBar().showMessage('鸟眼范围已'+('显示' if checked else '隐藏'),2000)

    def _on_af_toggled_from_view(self, checked: bool):
        self._show_af = checked
        if hasattr(self, '_a_tb_af'):
            self._a_tb_af.blockSignals(True); self._a_tb_af.setChecked(checked); self._a_tb_af.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_af']=checked; _save_prefs(_PREFS_PATH, p)

    def _on_yolo_bird_toggled_from_view(self, checked: bool):
        self._show_yolo_bird = checked
        if hasattr(self, '_a_tb_yolo_bird'):
            self._a_tb_yolo_bird.blockSignals(True); self._a_tb_yolo_bird.setChecked(checked); self._a_tb_yolo_bird.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_yolo_bird']=checked; _save_prefs(_PREFS_PATH, p)

    def _on_yolo_eye_toggled_from_view(self, checked: bool):
        self._show_yolo_eye = checked
        if hasattr(self, '_a_tb_yolo_eye'):
            self._a_tb_yolo_eye.blockSignals(True); self._a_tb_yolo_eye.setChecked(checked); self._a_tb_yolo_eye.blockSignals(False)
        p=_load_prefs(_PREFS_PATH); p['show_yolo_eye']=checked; _save_prefs(_PREFS_PATH, p)

    def _refresh_global_sort_btn(self):
        if not hasattr(self, '_tb_global_sort_btn'):
            return
        order = getattr(self.thumb_strip, '_sort_order', None)
        txt = {
            None: '⇅ 排序:名',
            'desc': '⇅ 排序:清',
            'asc': '⇅ 排序:清',
            'fdesc': '⇅ 排序:综',
            'fasc': '⇅ 排序:综',
        }.get(order, '⇅ 排序:名')
        self._tb_global_sort_btn.setText(txt)

    def _menu_set_sort(self, order: Optional[str]):
        if order not in (None, 'asc', 'desc', 'fasc', 'fdesc'):
            order = None
        if self._sort_by_burst_group and (not self._show_burst_groups) and (not self._burst_fold):
            self._sort_by_burst_group = False
            if hasattr(self, '_menu_a_sort_by_group'):
                self._menu_a_sort_by_group.blockSignals(True); self._menu_a_sort_by_group.setChecked(False); self._menu_a_sort_by_group.blockSignals(False)
            if hasattr(self, '_a_tb_sort_by_group'):
                self._a_tb_sort_by_group.blockSignals(True); self._a_tb_sort_by_group.setChecked(False); self._a_tb_sort_by_group.blockSignals(False)
            if hasattr(self.thumb_strip, 'set_sort_by_burst_group'):
                self.thumb_strip.set_sort_by_burst_group(False)
        self.thumb_strip._set_sort(order)
        for k, a in self._sort_actions.items():
            a.blockSignals(True); a.setChecked(str(order)==k); a.blockSignals(False)
        if hasattr(self, '_tb_sort_actions'):
            for k, a in self._tb_sort_actions.items():
                a.blockSignals(True); a.setChecked(str(order)==k); a.blockSignals(False)
        self._refresh_global_sort_btn()

    def _menu_set_filter_mode(self, mode: str):
        self.thumb_strip._set_filter_mode(mode)
        if hasattr(self, '_filter_mode_actions'):
            for k, a in self._filter_mode_actions.items():
                a.blockSignals(True); a.setChecked(k == mode); a.blockSignals(False)
        if mode == 'off' and hasattr(self, '_filter_actions'):
            for tier in ('green', 'yellow', 'red'):
                if tier in self._filter_actions:
                    self._filter_actions[tier].blockSignals(True)
                    self._filter_actions[tier].setChecked(True)
                    self._filter_actions[tier].blockSignals(False)

    def _menu_toggle_filter(self, tier: str, checked: bool):
        self.thumb_strip._toggle_filter(tier, checked)
        if tier in self._filter_actions:
            self._filter_actions[tier].blockSignals(True)
            self._filter_actions[tier].setChecked(checked)
            self._filter_actions[tier].blockSignals(False)

    def _menu_show_all_tiers(self):
        for tier in ('green','yellow','red'):
            self.thumb_strip._filter_tiers.add(tier)
            if tier in self._filter_actions:
                self._filter_actions[tier].blockSignals(True)
                self._filter_actions[tier].setChecked(True)
                self._filter_actions[tier].blockSignals(False)
        self.thumb_strip._rebuild_visible()
        self.thumb_strip._scroll_to(self.thumb_strip._current); self.thumb_strip.update()

    def open_file(self):
        path,_=QFileDialog.getOpenFileName(self,'打开 RAW 文件','',
            'RAW 文件 (*.nef *.cr2 *.cr3 *.arw *.orf *.NEF *.CR2 *.CR3 *.ARW *.ORF);;所有文件 (*)')
        if not path: return
        self._scan_dir(os.path.dirname(os.path.abspath(path)),open_path=path)

    def open_folder(self):
        d=QFileDialog.getExistingDirectory(self,'选择文件夹')
        if d: self._scan_dir(d)

    def reveal_current_in_explorer(self):
        path=self._current_path
        if not path:
            self.statusBar().showMessage('当前没有可定位的照片',3000)
            return
        p=str(Path(path).resolve())
        if not os.path.isfile(p):
            self.statusBar().showMessage('文件不存在，无法定位',4000)
            return
        folder=str(Path(p).parent)
        try:
            if sys.platform.startswith('win'):
                ok=False
                try:
                    import ctypes
                    from ctypes import wintypes
                    shell32=ctypes.windll.shell32
                    ole32=ctypes.windll.ole32
                    pidl_folder=ctypes.c_void_p()
                    pidl_item=ctypes.c_void_p()
                    attrs=wintypes.DWORD(0)
                    hr1=shell32.SHParseDisplayName(folder,None,ctypes.byref(pidl_folder),0,ctypes.byref(attrs))
                    hr2=shell32.SHParseDisplayName(p,None,ctypes.byref(pidl_item),0,ctypes.byref(attrs))
                    if hr1==0 and hr2==0 and pidl_folder and pidl_item:
                        arr=(ctypes.c_void_p*1)()
                        arr[0]=pidl_item
                        hr3=shell32.SHOpenFolderAndSelectItems(pidl_folder,1,arr,0)
                        ok=(hr3==0)
                    if pidl_item: ole32.CoTaskMemFree(pidl_item)
                    if pidl_folder: ole32.CoTaskMemFree(pidl_folder)
                except Exception:
                    ok=False
                if not ok:
                    os.startfile(folder)
            elif sys.platform=='darwin':
                subprocess.Popen(['open','-R',p])
            else:
                subprocess.Popen(['xdg-open',folder])
        except Exception as e:
            self.statusBar().showMessage(f'打开文件管理器失败：{e}',5000)

    def _scan_dir(self,folder:str,open_path:Optional[str]=None):
        retire_thread(self._score_thread); self._score_thread=None
        retire_thread(self._scan_thread); self._scan_thread=None
        self._preloader.clear()
        self._files=[]; self._idx=-1; self._current_path=None; self._scanning=True
        self._open_after_scan:Optional[str]=(str(Path(open_path).resolve()) if open_path else None)
        self._score_cache=ScoreCache(folder)
        self._flag_cache=FlagCache(folder)
        self._flag_cache.load()
        loaded=self._score_cache.load()
        n_cached=self._score_cache.count() if loaded else 0
        self._sb_batch.setText(_sb('CSV',f'已载入 {n_cached} 条',vc=SB_ET_OK) if loaded else '')
        self._sb_fused.setText('')
        self.thumb_strip.set_files([],None)
        self._sb_scan.setText(_sb('扫描','进行中…',vc=SB_VAL_WARN))
        self._sb_idx.setText(_sb('序号','—')); self._sb_size.setText('')
        self._sb_flag.setText('')
        self._refresh_nav_btns()
        t=ScanThread(folder)
        t.sig_batch.connect(self._on_scan_batch); t.sig_done.connect(self._on_scan_done)
        self._scan_thread=t; t.start()

    def _on_scan_batch(self,batch_paths:list,count:int):
        if not batch_paths: return
        prev_path=self._current_path
        self._files.extend(batch_paths)
        if prev_path:
            try: self._idx=self._files.index(prev_path)
            except ValueError: self._idx=0
        self._sb_scan.setText(_sb('扫描',f'{count} 个…',vc=SB_VAL_WARN))
        if self._idx<0 and self._files: self._show(0)
        self._refresh_nav_btns()

    def _on_scan_done(self,raw_paths:list):
        sorted_paths=list(raw_paths)
        prev_path=self._current_path; self._files=sorted_paths
        self._scanning=False; n=len(sorted_paths)
        if prev_path:
            try: self._idx=self._files.index(prev_path)
            except ValueError: self._idx=min(max(self._idx,0),n-1) if n>0 else -1
        open_tgt=getattr(self,'_open_after_scan',None)
        if open_tgt:
            tgt_lower=os.path.basename(open_tgt).lower()
            for i,f in enumerate(self._files):
                if os.path.basename(f).lower()==tgt_lower:
                    self._idx=-1; self._show(i); break
            self._open_after_scan=None
        if self._idx<0 and self._files: self._show(0)
        if n==0: self._sb_scan.setText(_sb('扫描','无 RAW 文件',vc=SB_VAL_ERR))
        else:    self._sb_scan.setText(_sb('共',f'{n} 个文件'))
        self._update_sb(); self._refresh_nav_btns()
        cur=max(0,self._idx)
        self.thumb_strip.set_files(self._files,self._score_cache,self._flag_cache,current_idx=cur,start_loader=False)
        if self._idx>=0: self.thumb_strip.set_current(self._idx)
        if self._score_cache:
            removed = self._score_cache.retain(self._files)
            if removed > 0:
                self._score_cache.save_if_dirty()
            total = len(self._files)
            to_score = self._score_cache.missing(self._files)
            to_fused = self._score_cache.missing_fused(self._files)
            missing_sharp = len(to_score)
            missing_fused = len(to_fused)
            cached_sharp  = total - missing_sharp
            cached_fused  = total - missing_fused

            if missing_sharp > 0 or missing_fused > 0:
                self._sb_batch.setText(_sb('CSV', f'{cached_sharp}/{total} 已缓存', vc=SB_VAL_WARN))
                if missing_fused > 0:
                    self._sb_fused.setText(_sb('综合', f'{cached_fused}/{total} 待评', vc=SB_VAL_WARN))
                self._start_combined_scoring(force_sharp=False, force_fused=False)
            else:
                self._sb_batch.setText(_sb('CSV', f'{cached_sharp}/{total} 已缓存', vc=SB_ET_OK))
                self._sb_fused.setText(_sb('综合', f'{cached_fused}/{total} 已缓存', vc=SB_ET_OK))
        if self._flag_cache:
            np_=self._flag_cache.count_pick(); nr=self._flag_cache.count_reject()
            if np_>0 or nr>0:
                self.statusBar().showMessage(f'已加载标记：★ 选用 {np_} 张 / ✕ 待删除 {nr} 张', 4000)

    def _start_combined_scoring(self, force_sharp: bool = False, force_fused: bool = False):
        """启动合并评分线程：单次 RAW 读取同时完成清晰度 + 融合评分。"""
        if not self._files or self._score_cache is None:
            return
        retire_thread(self._score_thread); self._score_thread = None

        # 强制重评时清除缓存
        if force_sharp:
            self._score_cache._data = {}
        if force_fused:
            for p in self._files:
                try: self._score_cache.set_fused(p, None)
                except Exception: pass

        # 统计待评数量，决定是否需要启动线程
        to_sharp = self._score_cache.missing(self._files)
        to_fused = self._score_cache.missing_fused(self._files)
        if not to_sharp and not to_fused:
            total = len(self._files)
            self._sb_batch.setText(_sb('CSV', f'全部 {total} 条已缓存', vc=SB_ET_OK))
            self._sb_fused.setText(_sb('综合', f'全部 {total} 条已缓存', vc=SB_ET_OK))
            return

        # 取待评路径的并集
        all_paths = list(dict.fromkeys(to_sharp + [p for p in to_fused if p not in set(to_sharp)]))
        total = len(self._files)
        self._sb_batch.setText(_sb('评分', f'{total - len(to_sharp)}/{total}', vc=SB_VAL_WARN))
        if to_fused:
            self._sb_fused.setText(_sb('综合', f'{total - len(to_fused)}/{total}', vc=SB_VAL_WARN))

        t = CombinedBatchScoringThread(all_paths, self._score_cache,
                                       force_sharp=force_sharp, force_fused=force_fused)
        t.sig_progress.connect(self._on_combined_progress)
        t.sig_done.connect(self._on_combined_done)
        self._score_thread = t
        t.start()

    def _on_combined_progress(self, filename: str, sharpness: float, fused: float, done: int, total: int):
        """合并评分线程进度回调——同时更新清晰度与融合分。"""
        all_total = len(self._files)
        if self._score_cache and all_total > 0:
            missing_sharp = len(self._score_cache.missing(self._files))
            missing_fused = len(self._score_cache.missing_fused(self._files))
            cached_sharp  = all_total - missing_sharp
            cached_fused  = all_total - missing_fused
            self._sb_batch.setText(_sb('评分', f'{cached_sharp}/{all_total}', vc=SB_VAL_WARN))
            if missing_fused > 0:
                self._sb_fused.setText(_sb('综合', f'{cached_fused}/{all_total}', vc=SB_VAL_WARN))

        self._batch_resume_retries = 0
        self._batch_last_cached    = all_total - (len(self._score_cache.missing(self._files)) if self._score_cache else 0)

        for i, p in enumerate(self._files):
            if os.path.basename(p) == filename:
                if sharpness > 0:
                    self.thumb_strip.update_score(i, sharpness, 'live')
                    if self._current_path and os.path.basename(self._current_path) == filename:
                        self.view.set_sharpness(sharpness, 'live')
                if fused > 0:
                    self.thumb_strip.update_fused_score(i, fused, 'live')
                    if self._current_path and os.path.basename(self._current_path) == filename:
                        self.view.set_fused_score(fused)
                break

    def _on_combined_done(self, scored: int):
        """合并评分线程完成回调。"""
        if not self._score_cache or not self._files:
            self._sb_batch.setText(''); self._sb_fused.setText('')
            return
        total         = len(self._files)
        missing_sharp = len(self._score_cache.missing(self._files))
        missing_fused = len(self._score_cache.missing_fused(self._files))
        cached_sharp  = total - missing_sharp
        cached_fused  = total - missing_fused

        if missing_sharp > 0:
            progressed = cached_sharp > self._batch_last_cached
            self._batch_last_cached   = cached_sharp
            self._batch_resume_retries = 0 if progressed else (self._batch_resume_retries + 1)
            if self._batch_resume_retries <= 3 and not self._scanning:
                self._sb_batch.setText(_sb('评分', f'{cached_sharp}/{total} 继续中', vc=SB_VAL_WARN))
                self.statusBar().showMessage(f'后台评分中断，自动续跑（第 {self._batch_resume_retries}/3 次）', 3500)
                QTimer.singleShot(600, lambda: self._start_combined_scoring(force_sharp=False, force_fused=False))
                return
            else:
                self._sb_batch.setText(_sb('评分', f'{cached_sharp}/{total} 已停止', vc=SB_VAL_ERR))
        else:
            self._batch_resume_retries = 0; self._batch_last_cached = -1
            self._sb_batch.setText(_sb('CSV', f'{cached_sharp}/{total} 已评', vc=SB_ET_OK))

        if missing_fused > 0:
            self._sb_fused.setText(_sb('综合', f'{cached_fused}/{total} 未完', vc=SB_VAL_WARN))
        else:
            self._sb_fused.setText(_sb('综合', f'{cached_fused}/{total} 已评', vc=SB_ET_OK))

        self.statusBar().showMessage(f'后台评分完成，新增 {scored} 张（基础分 + 综合分）', 4000)
        self.thumb_strip.ensure_sorted()

    def go_prev(self):
        self.thumb_strip.ensure_sorted()
        vis=self.thumb_strip._visible
        if not vis: return
        try:    vpos=vis.index(self._idx)
        except ValueError: vpos=0
        if vpos>0: self._show(vis[vpos-1])

    def go_next(self):
        self.thumb_strip.ensure_sorted()
        vis=self.thumb_strip._visible
        if not vis: return
        try:    vpos=vis.index(self._idx)
        except ValueError: vpos=len(vis)-1
        if vpos<len(vis)-1: self._show(vis[vpos+1])

    def _on_wheel_steps(self, dy: int):
        if not self._files or self._idx < 0: return
        self._wheel_accum += float(dy)
        if self._wheel_accum >= 0:
            steps = int(self._wheel_accum // 120)
        else:
            steps = -int((-self._wheel_accum) // 120)
        if steps:
            self._wheel_accum -= steps * 120
            self._wheel_pending_steps += steps
            self._wheel_timer.start(35)

    def _flush_wheel_nav(self):
        steps = self._wheel_pending_steps
        self._wheel_pending_steps = 0
        if not steps or not self._files or self._idx < 0: return
        self.thumb_strip.ensure_sorted()
        vis = self.thumb_strip._visible
        if not vis: return
        try:
            vpos = vis.index(self._idx)
        except ValueError:
            return
        target_vpos = max(0, min(len(vis)-1, vpos - steps))
        target_idx = vis[target_vpos]
        if target_idx != self._idx:
            self._show(target_idx)

    def _show(self,idx:int):
        if idx<0 or idx>=len(self._files): return
        retire_thread(self._thread)
        self._prev_zoom_mode=self.view.zoom_mode
        self._idx=idx; self._current_path=self._files[idx]
        path=self._current_path
        
        # 预检查缓存
        cached_data = self._preloader.get(path)
        
        fname=os.path.basename(path)
        dname=os.path.dirname(path)
        brand=BRAND_BY_EXT.get(Path(path).suffix.lower(),'?')
        ext=Path(path).suffix.upper()
        self._lbl_fname.setText(f'  [{brand} {ext}]  {dname}{os.sep}{fname}  ')
        self.view._ctx_fname_str=fname
        sfx='+' if self._scanning else ''
        self._sb_idx.setText(_sb('序号',f'{idx+1} / {len(self._files)}{sfx}'))
        self._sb_size.setText(_sb('尺寸','…')); self._sb_zoom.setText(_sb('缩放','…'))
        flag = self._flag_cache.get(path) if self._flag_cache else FLAG_NONE
        self.view.clear_yolo_overlays()
        self.view.set_flag(flag)
        self._refresh_flag_sb(flag)
        self._refresh_flag_toolbar(flag)
        self._refresh_nav_btns()
        
        self.thumb_strip.set_current(idx)
        if self._score_cache:
            cv=self._score_cache.get(path)
            if cv is not None and cv>0: self.view.set_sharpness(cv,'cache')
            
        if cached_data:
            # 命中缓存，直接显示，不进入 loading 状态，彻底消除黑屏
            self._on_load_done(cached_data)
        else:
            self.view.set_loading(True)
            t=LoadThread(path)
            t.sig_done.connect(self._on_load_done); t.sig_error.connect(self._on_load_error)
            self._thread=t; t.start()

    def _on_load_done(self,data:ImageData):
        FS=SB_FONT_SIZE
        if data.pixmap is None and data.qimage is not None and not data.qimage.isNull():
            data.pixmap = QPixmap.fromImage(data.qimage)
        self.view.load_pixmap(data.pixmap)
        self.view.set_af_info(data.af_points,data.ref_w,data.ref_h)
        self.view.set_exif_data(data.exif)
        self._current_qimage = data.qimage
        self._current_af_points = list(data.af_points or [])
        self._current_ref_w = int(data.ref_w or 0)
        self._current_ref_h = int(data.ref_h or 0)
        self._current_live_sharpness = float(data.sharpness or 0.0)

        if self._score_cache and self._current_path:
            fv = self._score_cache.get_fused(self._current_path)
            if fv is not None and float(fv) > 0:
                self.view.set_fused_score(float(fv))
                if self._idx >= 0:
                    self.thumb_strip.update_fused_score(self._idx, float(fv), 'cache')

        self._maybe_request_yolo(self._current_path, data.qimage)
        if self._current_path:
            yk = str(Path(self._current_path).resolve())
            if yk in self._yolo_cache:
                self._update_fused_for_current()

        flag = self._flag_cache.get(self._current_path) if self._flag_cache and self._current_path else FLAG_NONE
        self.view.set_flag(flag)
        if self._lock_zoom and self._prev_zoom_mode=='100': self.view.enter_100_at_af()
        self._sb_size.setText(_sb('尺寸',f'{data.disp_w} × {data.disp_h}'))
        live=float(data.sharpness or 0.0)
        effective = live
        if self._score_cache and self._current_path:
            cv = self._score_cache.get(self._current_path)
            if cv is None:
                self._score_cache.set(self._current_path, live, save_now=True)
            else:
                cv = float(cv or 0.0)
                denom = max(abs(cv), 1e-6)
                if abs(live - cv) / denom > SCORE_VERIFY_TOLERANCE:
                    self._score_cache.set(self._current_path, live, save_now=True)
                else:
                    effective = cv
            if self._idx>=0:
                self.thumb_strip.update_score(self._idx, effective, 'live')
        self.view.set_sharpness(effective,'live')
        self._update_sb()
        
        if self._files and not self.thumb_strip._thumbs:
            self.thumb_strip._start_loader(priority_idx=self._idx)
        self._trigger_preload(self._idx)

    def _maybe_request_yolo(self, path: Optional[str], qimage):
        if not path or not self._yolo or qimage is None or qimage.isNull():
            return
        key = str(Path(path).resolve())
        if key in self._yolo_cache or path in self._yolo_cache:
            r = self._yolo_cache.get(key) or self._yolo_cache.get(path) or {}
            self.view.set_yolo_overlays(r.get('bird_boxes'), r.get('bird_polys'), r.get('eye_boxes'))
            return
        self._yolo.request(path, qimage)

    def _update_fused_for_current(self):
        if compute_fused_score is None:
            self.view.set_fused_score(0.0)
            return
        if not self._current_path or self._current_qimage is None or self._current_qimage.isNull():
            self.view.set_fused_score(0.0)
            return
        key = str(Path(self._current_path).resolve())
        y = self._yolo_cache.get(key) or self._yolo_cache.get(self._current_path) or {}
        has_yolo = bool(y) and bool(y.get('bird_boxes') or y.get('bird_polys') or y.get('eye_boxes'))
        if not has_yolo:
            if self._score_cache and self._current_path:
                fv = self._score_cache.get_fused(self._current_path)
                if fv is not None and float(fv) > 0:
                    self.view.set_fused_score(float(fv))
                    if self._idx >= 0:
                        self.thumb_strip.update_fused_score(self._idx, float(fv), 'cache')
                    return
            self.view.set_fused_score(0.0)
            return
        try:
            ret = compute_fused_score(
                self._current_qimage,
                self._current_af_points,
                self._current_ref_w,
                self._current_ref_h,
                self._current_live_sharpness,
                y,
            )
            if not ret:
                self.view.set_fused_score(0.0)
                return
            score, _details = ret
        except Exception:
            self.view.set_fused_score(0.0)
            return

        self._fused_cache[key] = float(score or 0.0)
        self.view.set_fused_score(score)
        if self._score_cache and self._current_path:
            bird_box_str = ""
            if y.get('bird_boxes'):
                b = y['bird_boxes'][0]
                bird_box_str = f"{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}"
            self._score_cache.set_fused(self._current_path, score, bird_box=bird_box_str)
        if self._idx >= 0:
            self.thumb_strip.update_fused_score(self._idx, score, 'live')

    def _on_yolo_ready(self, path: str, result: dict):
        if not path:
            return
        key = str(Path(path).resolve())
        self._yolo_cache[key] = result or {}
        # LRU 淘汰：防止大目录下 yolo_cache 无限增长（每条含多边形坐标，内存可观）
        _MAX_YOLO_CACHE = 40
        if len(self._yolo_cache) > _MAX_YOLO_CACHE:
            try:
                oldest = next(iter(self._yolo_cache))
                del self._yolo_cache[oldest]
            except Exception:
                pass
        if self._current_path and str(Path(self._current_path).resolve()) == key:
            r = self._yolo_cache.get(key) or {}
            self.view.set_yolo_overlays(r.get('bird_boxes'), r.get('bird_polys'), r.get('eye_boxes'))
            self._update_fused_for_current()

    def _trigger_preload(self, idx: int):
        if idx < 0 or not self._files: return
        vis = self.thumb_strip._visible
        if not vis: return
        try:
            vpos = vis.index(idx)
        except ValueError:
            return

        to_preload = []
        # 预加载顺序：下一张 -> 上一张 -> 下下张 -> 前前张 -> 下下下张
        # 优先加载用户最可能点击的方向（通常是向后浏览）
        indices = [vpos + 1, vpos - 1, vpos + 2, vpos - 2, vpos + 3]
        for i in indices:
            if 0 <= i < len(vis):
                p = self._files[vis[i]]
                if p not in to_preload:
                    to_preload.append(p)
        
        if to_preload:
            self._preloader.preload(to_preload)

    def _drop_unreadable_current(self, msg: str):
        path = self._current_path
        if not path or path not in self._files:
            self.view.set_loading(False); self.view.load_pixmap(None)
            self.statusBar().showMessage(f'加载失败：{msg}', 8000)
            return
        bad_name = os.path.basename(path)
        old_idx = self._idx if self._idx >= 0 else 0
        if self._score_cache:
            self._score_cache.remove(path); self._score_cache.save_if_dirty()
        if self._flag_cache:
            self._flag_cache.remove(path); self._flag_cache.save_if_dirty()
        self._preloader.remove(path)
        self._files = [p for p in self._files if p != path]
        self.view.set_loading(False)
        if not self._files:
            self._idx = -1; self._current_path = None
            self._lbl_fname.setText('  未打开文件  ')
            self._sb_idx.setText(_sb('序号', '—')); self._sb_size.setText('')
            self._sb_flag.setText('')
            self.thumb_strip.set_files([], self._score_cache, self._flag_cache, current_idx=0, start_loader=False)
            self.view.load_pixmap(None)
            self._refresh_nav_btns()
            self.statusBar().showMessage(f'已跳过不可读文件：{bad_name}', 8000)
            return
        next_idx = max(0, min(old_idx, len(self._files) - 1))
        self.thumb_strip.set_files(self._files, self._score_cache, self._flag_cache, current_idx=next_idx, start_loader=False)
        self._idx = -1; self._current_path = None
        self.statusBar().showMessage(f'已跳过不可读文件：{bad_name}（{msg}）', 8000)
        self._show(next_idx)

    def _on_load_error(self,msg:str):
        self._drop_unreadable_current(msg)

    def _update_sb(self):
        FS=SB_FONT_SIZE
        if self._files and self._idx>=0:
            sfx='+' if self._scanning else ''
            self._sb_idx.setText(_sb('序号',f'{self._idx+1} / {len(self._files)}{sfx}'))
        is_100=self.view.zoom_mode=='100'
        self._sb_zoom.setText(_sb('缩放','100%' if is_100 else '适合窗口',
                                  vc=SB_VAL_ACTIVE if is_100 else SB_VAL))
        self._sb_zoom.setStyleSheet(f'font-size:{FS}px;padding:0 6px;')

    def _refresh_nav_btns(self):
        en=bool(self._files) and self._idx>=0
        if hasattr(self,'_a_prev'):
            vis=self.thumb_strip._visible if hasattr(self,'thumb_strip') else []
            try:    vpos=vis.index(self._idx)
            except ValueError: vpos=-1
            self._a_prev.setEnabled(vpos>0)
            self._a_next.setEnabled(vpos<len(vis)-1)
        can_delete = en and HAS_SEND2TRASH
        if hasattr(self,'_a_delete'):      self._a_delete.setEnabled(can_delete)
        if hasattr(self,'_menu_a_delete'): self._menu_a_delete.setEnabled(can_delete)

    def _show_delete_blocked(self):
        txt = '未安装 send2trash，已禁止删除以避免直接永久删除。\n\n请先在当前虚拟环境安装：\npython -m pip install send2trash'
        QMessageBox.critical(self, '删除被阻止', txt)
        self.statusBar().showMessage('删除被阻止：未安装 send2trash', 6000)

    def delete_current_file(self):
        if not HAS_SEND2TRASH:
            self._show_delete_blocked(); return
        if not self._files or self._idx<0: return
        path=str(Path(self._files[self._idx]).resolve())
        self._files[self._idx]=path
        fname=os.path.basename(path)
        if self._confirm_delete:
            tn=delete_hint_text(HAS_SEND2TRASH)
            msg=QMessageBox(self); msg.setWindowTitle('确认删除')
            msg.setText(f'确定要删除以下文件吗？\n\n  {fname}\n\n{tn}')
            if _PYSIDE6:
                msg.setIcon(QMessageBox.Icon.Warning)
                _Yes=QMessageBox.StandardButton.Yes; _No=QMessageBox.StandardButton.No
            else:
                msg.setIcon(QMessageBox.Warning); _Yes=QMessageBox.Yes; _No=QMessageBox.No
            msg.setStandardButtons(_Yes|_No); msg.setDefaultButton(_No)
            cb=QCheckBox('不再询问（可通过“设置”重新开启）'); msg.setCheckBox(cb)
            reply=msg.exec() if _PYSIDE6 else msg.exec_()
            if cb.isChecked():
                self._a_confirm_del.blockSignals(True); self._a_confirm_del.setChecked(False)
                self._a_confirm_del.blockSignals(False)
                self._confirm_delete=False; self._a_confirm_del.setText('询删:关')
                p=_load_prefs(_PREFS_PATH); p['confirm_delete']=False; _save_prefs(_PREFS_PATH, p)
            if reply!=_Yes: return
        self._do_delete([path])

    def _delete_paths(self,paths:List[str]):
        if not HAS_SEND2TRASH:
            self._show_delete_blocked(); return
        if not paths: return
        n=len(paths)
        if self._confirm_delete:
            msg=QMessageBox(self); msg.setWindowTitle('确认删除')
            if n==1: detail=f'  {os.path.basename(paths[0])}'
            else:
                names='\n'.join(f'  {os.path.basename(p)}' for p in paths[:6])
                if n>6: names+=f'\n  … 共 {n} 个文件'
                detail=names
            tn=delete_hint_text(HAS_SEND2TRASH)
            msg.setText(f'确定要删除 {"文件" if n==1 else str(n)+" 个文件"}？\n\n{detail}\n\n{tn}')
            if _PYSIDE6:
                msg.setIcon(QMessageBox.Icon.Warning)
                _Yes=QMessageBox.StandardButton.Yes; _No=QMessageBox.StandardButton.No
            else:
                msg.setIcon(QMessageBox.Warning); _Yes=QMessageBox.Yes; _No=QMessageBox.No
            msg.setStandardButtons(_Yes|_No); msg.setDefaultButton(_No)
            if (msg.exec() if _PYSIDE6 else msg.exec_())!=_Yes: return
        self._do_delete(paths)

    def _do_delete(self, paths: List[str]):
        retire_thread(self._thread); self._thread = None
        retire_thread(self._score_thread); self._score_thread = None
        path_set = {str(Path(p).resolve()) for p in paths}
        deleted = []; was_current = self._current_path

        vis_paths_before: List[str] = [
            self._files[i] for i in self.thumb_strip._visible
            if 0 <= i < len(self._files)
        ]
        try:
            vis_pos_before = vis_paths_before.index(was_current)
        except ValueError:
            vis_pos_before = 0

        for path in path_set:
            try:
                move_to_trash_strict(path, HAS_SEND2TRASH, _send2trash)
                deleted.append(path)
                if self._score_cache: self._score_cache.remove(path)
                if self._flag_cache:  self._flag_cache.remove(path)
                self._preloader.remove(path)
                xp = _xmp_path(path)
                if os.path.isfile(xp):
                    try:
                        move_to_trash_strict(xp, HAS_SEND2TRASH, _send2trash)
                    except: pass
            except Exception as e:
                self.statusBar().showMessage(f'删除失败：{os.path.basename(path)}: {e}', 5000)
        if self._score_cache and deleted: self._score_cache.save()
        if self._flag_cache  and deleted: self._flag_cache.save()
        if not deleted: return
        self.statusBar().showMessage(
            f'已移入回收站 {len(deleted)} 个文件', 4000)

        self._files = [f for f in self._files if f not in deleted]
        cur_hint = max(0, min(self._idx, len(self._files) - 1)) if self._files else 0
        self.thumb_strip.set_files(self._files, self._score_cache, self._flag_cache,
                                   current_idx=cur_hint)

        if not self._files:
            self._idx = -1; self._current_path = None
            self._lbl_fname.setText('  未打开文件  ')
            self._sb_idx.setText(_sb('序号', '—')); self._sb_size.setText('')
            self._sb_flag.setText('')
            self.view.load_pixmap(None); self._refresh_nav_btns(); return

        if was_current in deleted:
            next_path = None
            for p in vis_paths_before[vis_pos_before + 1:]:
                if p not in path_set and p in self._files:
                    next_path = p; break
            if next_path is None:
                for p in reversed(vis_paths_before[:vis_pos_before]):
                    if p not in path_set and p in self._files:
                        next_path = p; break
            if next_path is None and self.thumb_strip._visible:
                first_vis = self.thumb_strip._visible[0]
                if 0 <= first_vis < len(self._files):
                    next_path = self._files[first_vis]

            if next_path is not None:
                try:
                    next_idx = self._files.index(next_path)
                    self._idx = -1; self._show(next_idx)
                except ValueError:
                    self._idx = -1; self.view.load_pixmap(None)
            else:
                self._idx = -1; self._current_path = None
                self.view.load_pixmap(None)
        else:
            try:
                self._idx = self._files.index(was_current)
                self.thumb_strip.set_current(self._idx)
            except ValueError:
                ni = min(self._idx, len(self._files) - 1)
                self._idx = -1; self._show(ni)
        self._refresh_nav_btns()
        # 删除后重启评分线程（之前 retire_thread 已将其停止）
        self._start_combined_scoring(force_sharp=False, force_fused=False)

    def closeEvent(self,e):
        # 先停止 YOLO overlay 线程池，再停其他线程，避免关闭时析构崩溃
        if self._yolo:
            try: self._yolo.shutdown()
            except Exception: pass
        retire_thread(self._score_thread)
        retire_thread(self._thread)
        retire_thread(self._scan_thread)
        self._preloader.clear()
        self.thumb_strip.stop_loader()
        cleanup_zombies_force()
        if _ZOMBIES:
            for t in list(_ZOMBIES): t.wait(500)
        if self._score_cache: self._score_cache.save_if_dirty()
        if self._flag_cache:  self._flag_cache.save_if_dirty()
        super().closeEvent(e)


# ════════════════════════════════════════════════════════════════════
#  § 辅助 UI
# ════════════════════════════════════════════════════════════════════

def _msep() -> QLabel:
    l=QLabel('│'); l.setStyleSheet(f'color:{SB_SEP};font-size:15px;padding:0 2px;')
    l.setFocusPolicy(_NO_FOCUS)
    return l

def _apply_dark_theme(app):
    app.setStyle('Fusion'); pal=QPalette()
    pal.setColor(QPalette.Window,          QColor( 43, 43, 43))
    pal.setColor(QPalette.WindowText,      QColor(200,200,200))
    pal.setColor(QPalette.Base,            QColor( 36, 36, 36))
    pal.setColor(QPalette.AlternateBase,   QColor( 48, 48, 48))
    pal.setColor(QPalette.ToolTipBase,     QColor( 48, 48, 48))
    pal.setColor(QPalette.ToolTipText,     QColor(220,220,220))
    pal.setColor(QPalette.Text,            QColor(200,200,200))
    pal.setColor(QPalette.BrightText,      QColor(230,230,230))
    pal.setColor(QPalette.Button,          QColor( 52, 52, 52))
    pal.setColor(QPalette.ButtonText,      QColor(200,200,200))
    pal.setColor(QPalette.Highlight,       QColor(110,110,110))
    pal.setColor(QPalette.HighlightedText, QColor(245,245,245))
    pal.setColor(QPalette.Disabled,QPalette.ButtonText, QColor( 62, 68, 84))
    pal.setColor(QPalette.Disabled,QPalette.WindowText, QColor( 62, 68, 84))
    app.setPalette(pal)


# ════════════════════════════════════════════════════════════════════
#  § 程序入口
# ════════════════════════════════════════════════════════════════════

def main():
    if hasattr(QApplication, 'setHighDpiScaleFactorRoundingPolicy'):
        try:
            policy = Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        except AttributeError:
            policy = 0  # PassThrough value on PyQt5
        QApplication.setHighDpiScaleFactorRoundingPolicy(policy)
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_build_windows_appid())
        except Exception:
            pass
    app=QApplication(sys.argv)
    app.setApplicationName('RAW Photo Viewer'); app.setOrganizationName('RAWViewer')
    icon_path = _resolve_app_icon()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    _apply_dark_theme(app)
    app.setFont(QFont(UI_FONT_FAMILY, UI_FONT_BASE_SIZE))
    win=MainWindow()
    if icon_path:
        win.setWindowIcon(QIcon(icon_path))
    win.showMaximized()
    if len(sys.argv)>1:
        arg=os.path.abspath(sys.argv[1])
        if os.path.isfile(arg):
            ext=Path(arg).suffix.lower()
            if ext in SUPPORTED_EXTS: win._scan_dir(os.path.dirname(arg),open_path=arg)
            else: print(f"不支持的格式: {ext}")
        elif os.path.isdir(arg): win._scan_dir(arg)
        else: print(f"路径不存在: {arg}")
    sys.exit(app.exec())


if __name__=='__main__':
    main()