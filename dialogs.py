#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对话框：RejectPreviewDialog（待删除预览）和 SettingsDialog（设置）。"""

import os
from typing import List, Optional

try:
    import send2trash; HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

from qt_compat import (
    QDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QAbstractItemView, QFrame, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QCheckBox, QSlider, QPixmap,
    QColor, QThread, Signal, Qt,
    _PYSIDE6, _ALIGN_CENTER, _KEEP_AR, _SMOOTH_XFORM,
)
from utils import delete_hint_text, delete_hint_short_text, FlagCache, ScoreCache, SHARP_HI, SHARP_LO
from raw_reader import RAWReader
from workers import retire_thread


# ════════════════════════════════════════════════════════════════════
#  § 对话框共用样式
# ════════════════════════════════════════════════════════════════════

_DLGSTY = '''
    QDialog { background: #0d0e12; }
    QLabel#title_label { color:#a06070; font-size:15px; font-weight:bold; padding:0; }
    QLabel#subtitle_label { color:#525c70; font-size:12px; }
    QListWidget {
        background:#0a0b0f; border:1px solid #1e2029; border-radius:6px;
        color:#a8b0c4; font-size:13px; outline:none; }
    QListWidget::item { padding:8px 12px; border-bottom:1px solid #111318; }
    QListWidget::item:selected { background:#1a2035; color:#a06070; }
    QListWidget::item:hover:!selected { background:#111318; }
    QLabel#preview_placeholder {
        color:#3a4050; font-size:13px;
        background:#0a0b0f; border:1px solid #1e2029; border-radius:6px; }
    QLabel#preview_fname { color:#a8b0c4; font-size:13px;
        padding:6px 10px 2px 10px; background:transparent; }
    QLabel#preview_meta { color:#525c70; font-size:12px;
        padding:0 10px 6px 10px; background:transparent; }
    QPushButton { font-size:13px; padding:7px 18px; border-radius:6px; border:none; }
    QPushButton#btn_unmark { background:#142a20; color:#6db88a; border:1px solid #1e4030; }
    QPushButton#btn_unmark:hover { background:#1e4030; }
    QPushButton#btn_unmark:disabled { background:#111318; color:#3a4050; border-color:#1e2029; }
    QPushButton#btn_del_sel { background:#22141a; color:#b06878; border:1px solid #3a1c26; }
    QPushButton#btn_del_sel:hover { background:#3a1c26; }
    QPushButton#btn_del_sel:disabled { background:#111318; color:#3a4050; border-color:#1e2029; }
    QPushButton#btn_del_all { background:#5a2838; color:#e0c0c8; }
    QPushButton#btn_del_all:hover { background:#6e3044; }
    QPushButton#btn_close { background:#16181f; color:#525c70; }
    QPushButton#btn_close:hover { background:#1e2029; }
    QScrollBar:vertical { background:#0a0b0f; width:8px; border-radius:4px; }
    QScrollBar::handle:vertical { background:#1e2029; border-radius:4px; min-height:20px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
'''


# ════════════════════════════════════════════════════════════════════
#  § 预览 Label（自适应缩放）
# ════════════════════════════════════════════════════════════════════

class PreviewPixmapLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pm: Optional[QPixmap] = None
        self.setMinimumSize(300, 200)
        self.setAlignment(_ALIGN_CENTER)
        self.setStyleSheet('background:#0d0e12;border:1px solid #111318;border-radius:6px;')

    def set_pixmap(self, pm: Optional[QPixmap]):
        self._source_pm = pm; self._refresh()

    def _refresh(self):
        if self._source_pm is None or self._source_pm.isNull(): self.clear(); return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0: return
        super().setPixmap(self._source_pm.scaled(w, h, _KEEP_AR, _SMOOTH_XFORM))

    def resizeEvent(self, e):
        super().resizeEvent(e); self._refresh()


# ════════════════════════════════════════════════════════════════════
#  § 预览加载线程
# ════════════════════════════════════════════════════════════════════

class RejectPreviewLoadThread(QThread):
    sig_done  = Signal(object, str)
    sig_error = Signal(str, str)

    def __init__(self, path: str):
        super().__init__(); self._path = path; self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        path = self._path
        if self._cancelled: return
        qi, err = RAWReader.read_qimage(path)
        if self._cancelled: return
        if qi and not qi.isNull(): self.sig_done.emit(qi, path)
        else:                      self.sig_error.emit(err or '读取失败', path)


# ════════════════════════════════════════════════════════════════════
#  § 待删除照片预览对话框
# ════════════════════════════════════════════════════════════════════

class RejectPreviewDialog(QDialog):
    sig_unmark = Signal(str)
    sig_delete = Signal(list)

    def __init__(self, reject_paths: List[str], flag_cache: FlagCache,
                 parent=None, score_cache: Optional[ScoreCache] = None):
        super().__init__(parent)
        self.setWindowTitle(f'待删除照片预览  （共 {len(reject_paths)} 张）')
        self.setMinimumSize(900, 580); self.resize(1060, 640); self.setModal(True)
        self.setStyleSheet(_DLGSTY)

        self._paths       = list(reject_paths)
        self._flag_cache  = flag_cache
        self._score_cache = score_cache
        self._load_thread: Optional[RejectPreviewLoadThread] = None
        self._current_preview_path: Optional[str] = None

        self._build_ui()
        if self._list_widget.count() > 0: self._list_widget.setCurrentRow(0)

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(18,16,18,14); root.setSpacing(12)
        hdr = QHBoxLayout(); hdr.setSpacing(10)
        self._title_lbl = QLabel(f'✕  待删除照片  —  共 {len(self._paths)} 张')
        self._title_lbl.setObjectName('title_label')
        hint = QLabel(delete_hint_short_text(HAS_SEND2TRASH)); hint.setObjectName('subtitle_label')
        hdr.addWidget(self._title_lbl); hdr.addStretch(1); hdr.addWidget(hint)
        root.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine if _PYSIDE6 else QFrame.HLine)
        sep.setStyleSheet('color:#1a1c24;'); root.addWidget(sep)

        splitter = QSplitter(Qt.Orientation.Horizontal if _PYSIDE6 else Qt.Horizontal)
        splitter.setHandleWidth(4); splitter.setStyleSheet('QSplitter::handle{background:#1a1c24;}')

        left = QWidget(); llay = QVBoxLayout(left)
        llay.setContentsMargins(0,0,0,0); llay.setSpacing(4)
        llay.addWidget(QLabel('文件列表') if True else None)
        self._list_widget = QListWidget()
        self._list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection if _PYSIDE6
            else QAbstractItemView.SingleSelection)
        for path in self._paths:
            item = QListWidgetItem(f'  ✕  {os.path.basename(path)}')
            item.setData(Qt.ItemDataRole.UserRole if _PYSIDE6 else Qt.UserRole, path)
            item.setForeground(QColor('#b06070'))
            self._list_widget.addItem(item)
        self._list_widget.currentItemChanged.connect(self._on_list_select)
        llay.addWidget(self._list_widget); splitter.addWidget(left)

        right = QWidget(); rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0,0,0,0); rlay.setSpacing(0)
        self._preview_lbl = PreviewPixmapLabel()
        self._preview_lbl.setObjectName('preview_placeholder')
        self._preview_lbl.setText('← 点击左侧文件名预览')
        self._fname_lbl = QLabel(''); self._fname_lbl.setObjectName('preview_fname')
        self._meta_lbl  = QLabel(''); self._meta_lbl.setObjectName('preview_meta')
        rlay.addWidget(self._preview_lbl, 1)
        rlay.addWidget(self._fname_lbl); rlay.addWidget(self._meta_lbl)
        splitter.addWidget(right); splitter.setSizes([260, 720])
        root.addWidget(splitter, 1)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self._btn_unmark  = QPushButton('○  取消此张标记'); self._btn_unmark.setObjectName('btn_unmark')
        self._btn_del_sel = QPushButton('🗑  删除此张');      self._btn_del_sel.setObjectName('btn_del_sel')
        self._btn_del_all = QPushButton(f'🗑  全部删除（{len(self._paths)} 张）')
        self._btn_del_all.setObjectName('btn_del_all')
        btn_close = QPushButton('关闭'); btn_close.setObjectName('btn_close')
        self._btn_unmark.setEnabled(False); self._btn_del_sel.setEnabled(False)
        self._btn_unmark.setToolTip('将当前预览的照片从"待删除"中移除，恢复为未标记')
        self._btn_del_sel.setToolTip('删除当前预览的照片')
        self._btn_del_all.setToolTip('删除列表中所有待删除照片')
        self._btn_unmark.clicked.connect(self._on_unmark)
        self._btn_del_sel.clicked.connect(self._on_delete_selected)
        self._btn_del_all.clicked.connect(self._on_delete_all)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_unmark); btn_row.addWidget(self._btn_del_sel)
        btn_row.addStretch(1); btn_row.addWidget(self._btn_del_all); btn_row.addWidget(btn_close)
        root.addLayout(btn_row); self._refresh_count_texts()

    def _refresh_count_texts(self):
        n = len(self._paths)
        self.setWindowTitle(f'待删除照片预览  （共 {n} 张）')
        self._title_lbl.setText(f'✕  待删除照片  —  共 {n} 张')
        self._btn_del_all.setText(f'🗑  全部删除（{n} 张）')
        self._btn_del_all.setEnabled(n > 0)

    def _on_list_select(self, current, _previous):
        if current is None:
            self._current_preview_path = None; self._preview_lbl.set_pixmap(None)
            self._preview_lbl.setText('← 点击左侧文件名预览')
            self._fname_lbl.setText(''); self._meta_lbl.setText('')
            self._btn_unmark.setEnabled(False); self._btn_del_sel.setEnabled(False); return
        path = current.data(Qt.ItemDataRole.UserRole if _PYSIDE6 else Qt.UserRole)
        self._current_preview_path = path
        self._btn_unmark.setEnabled(True); self._btn_del_sel.setEnabled(True)
        self._fname_lbl.setText(os.path.basename(path))
        self._meta_lbl.setText('加载中…')
        self._preview_lbl.set_pixmap(None); self._preview_lbl.setText('加载中，请稍候…')
        retire_thread(self._load_thread)
        t = RejectPreviewLoadThread(path)
        t.sig_done.connect(self._on_preview_loaded); t.sig_error.connect(self._on_preview_error)
        self._load_thread = t; t.start()

    def _on_preview_loaded(self, qi, path: str):
        if path != self._current_preview_path: return
        pm = QPixmap.fromImage(qi) if qi is not None and not qi.isNull() else None
        self._preview_lbl.set_pixmap(pm); self._preview_lbl.setText('')
        size_str = f'{pm.width()} × {pm.height()}' if pm else ''
        score_str = ''
        if self._score_cache:
            sv = self._score_cache.get(path)
            if sv is not None and sv > 0:
                tier = '🟢 清晰' if sv >= SHARP_HI else ('🟡 中等' if sv >= SHARP_LO else '🔴 偏软')
                score_str = f'   基础分：{int(sv)}  {tier}'
        try: fsize_str = f'{os.path.getsize(path)/1024/1024:.1f} MB'
        except: fsize_str = ''
        meta_parts = [size_str]
        if fsize_str: meta_parts.append(fsize_str)
        if score_str: meta_parts.append(score_str)
        self._fname_lbl.setText(os.path.basename(path))
        self._meta_lbl.setText('  ·  '.join(meta_parts))

    def _on_preview_error(self, err: str, path: str):
        if path != self._current_preview_path: return
        self._preview_lbl.set_pixmap(None); self._preview_lbl.setText(f'预览加载失败\n{err}')
        self._meta_lbl.setText(err)

    def _on_unmark(self):
        path = self._current_preview_path
        if not path: return
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole if _PYSIDE6 else Qt.UserRole) == path:
                self._list_widget.takeItem(row); break
        self._paths = [p for p in self._paths if p != path]
        self.sig_unmark.emit(path); self._refresh_count_texts()
        if self._list_widget.count() == 0:
            self._current_preview_path = None; self._preview_lbl.set_pixmap(None)
            self._preview_lbl.setText('列表已清空'); self._fname_lbl.setText('')
            self._meta_lbl.setText(''); self._btn_unmark.setEnabled(False)
            self._btn_del_sel.setEnabled(False)

    def _on_delete_selected(self):
        path = self._current_preview_path
        if not path: return
        self._remove_path_from_list(path); self.sig_delete.emit([path])

    def _on_delete_all(self):
        if not self._paths: return
        msg = QMessageBox(self); msg.setWindowTitle('确认全部删除')
        msg.setText(f'确定要删除全部 {len(self._paths)} 张"待删除"照片？\n\n{delete_hint_text(HAS_SEND2TRASH)}')
        if _PYSIDE6:
            msg.setIcon(QMessageBox.Icon.Warning)
            _Yes = QMessageBox.StandardButton.Yes; _No = QMessageBox.StandardButton.No
        else:
            msg.setIcon(QMessageBox.Warning); _Yes = QMessageBox.Yes; _No = QMessageBox.No
        msg.setStandardButtons(_Yes | _No); msg.setDefaultButton(_No)
        if msg.exec() == _Yes:
            self.sig_delete.emit(list(self._paths)); self.accept()

    def _remove_path_from_list(self, path: str):
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole if _PYSIDE6 else Qt.UserRole) == path:
                self._list_widget.takeItem(row); break
        self._paths = [p for p in self._paths if p != path]; self._refresh_count_texts()
        if self._list_widget.count() == 0:
            self._preview_lbl.set_pixmap(None); self._preview_lbl.setText('列表已清空')
            self._fname_lbl.setText(''); self._meta_lbl.setText('')
            self._btn_unmark.setEnabled(False); self._btn_del_sel.setEnabled(False)

    def closeEvent(self, e):
        retire_thread(self._load_thread); super().closeEvent(e)


# ════════════════════════════════════════════════════════════════════
#  § 设置对话框
# ════════════════════════════════════════════════════════════════════

_SLIDER_STYLE = '''
    QSlider::groove:horizontal {
        height: 4px; background: #1e2029;
        border-radius: 2px; border: 1px solid #2a2d3a;
    }
    QSlider::sub-page:horizontal {
        background: #4a78b0; border-radius: 2px;
    }
    QSlider::add-page:horizontal {
        background: #1e2029; border-radius: 2px;
    }
    QSlider::handle:horizontal {
        width: 14px; height: 14px; margin: -6px 0;
        border-radius: 7px;
        background: #6a98d0; border: 1px solid #3a5880;
    }
    QSlider::handle:horizontal:hover {
        background: #88b4e8;
    }
'''


class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('设置')
        self.setMinimumWidth(480)
        self.setModal(True)
        self.setStyleSheet('''
            QDialog   { background:#111318; }
            QGroupBox { color:#525c70; font-size:13px; font-weight:600;
                        border:1px solid #1e2029; border-radius:8px;
                        margin-top:12px; padding-top:10px; }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 4px; }
            QCheckBox { color:#a8b0c4; font-size:14px; spacing:8px; }
            QCheckBox::indicator { width:16px; height:16px; border-radius:3px;
                                   border:1px solid #3a4050; background:#1a1c24; }
            QCheckBox::indicator:checked { background:#4a78b0; border-color:#4a78b0; }
            QLabel    { color:#525c70; font-size:13px; }
            QPushButton { font-size:14px; padding:7px 22px; border-radius:6px; border:none; }
        ''')
        p = parent
        lay = QVBoxLayout(self)
        lay.setSpacing(14)
        lay.setContentsMargins(18, 18, 18, 18)

        # ── 文件操作 ──────────────────────────────────────────────────
        g1 = QGroupBox('文件操作'); v1 = QVBoxLayout(g1); v1.setSpacing(10)
        self._cb_confirm = QCheckBox('删除文件前弹窗确认')
        self._cb_confirm.setChecked(p._confirm_delete)
        v1.addWidget(self._cb_confirm)
        v1.addWidget(QLabel('提示：' + delete_hint_text(HAS_SEND2TRASH)))
        lay.addWidget(g1)

        # ── 查看 ──────────────────────────────────────────────────────
        g2 = QGroupBox('查看'); v2 = QVBoxLayout(g2); v2.setSpacing(10)
        self._cb_lock       = QCheckBox('锁定缩放（翻页时保持当前缩放级别）')
        self._cb_lock.setChecked(p._lock_zoom)
        self._cb_exif       = QCheckBox('显示 EXIF 信息覆盖层')
        self._cb_exif.setChecked(p._show_exif)
        self._cb_group_head = QCheckBox('连拍分组组号仅组首显示')
        self._cb_group_head.setChecked(p._burst_label_head_only)
        v2.addWidget(self._cb_lock)
        v2.addWidget(self._cb_exif)
        v2.addWidget(self._cb_group_head)
        lay.addWidget(g2)

        # ── YOLO 检测阈值 ─────────────────────────────────────────────
        g5 = QGroupBox('YOLO 检测阈值')
        v5 = QVBoxLayout(g5); v5.setSpacing(12)
        v5.addWidget(QLabel(
            '调低阈值可检测更多目标（误检增加），调高则仅保留高置信度结果。'))

        # 鸟身阈值
        bird_init = int(round(getattr(p, '_yolo_bird_conf', 0.35) * 100))
        self._sld_bird, self._lbl_bird = self._make_slider_row(
            v5, '鸟身轮廓（Segmentation）', bird_init, 5, 95)

        # 鸟眼阈值
        eye_init = int(round(getattr(p, '_yolo_eye_conf', 0.30) * 100))
        self._sld_eye, self._lbl_eye = self._make_slider_row(
            v5, '鸟眼检测（Detection）', eye_init, 5, 95)

        lay.addWidget(g5)

        # ── 基础分缓存 ────────────────────────────────────────────────
        g3 = QGroupBox('基础分缓存 (CSV)'); v3 = QVBoxLayout(g3); v3.setSpacing(6)
        cache = p._score_cache
        if cache:
            st = QLabel(f'📄 {cache.csv_path}\n已缓存 {cache.count()} 个文件的评分')
            st.setStyleSheet('color:#6db88a;font-size:13px;')
        else:
            st = QLabel('（尚未打开目录）')
            st.setStyleSheet('color:#525c70;font-size:13px;')
        st.setWordWrap(True)
        v3.addWidget(st)
        self._btn_rescore = QPushButton('🔄 强制重新评分全部文件')
        self._btn_rescore.setStyleSheet(
            'background:#1e2029;color:#a8b0c4;font-size:13px;padding:5px 12px;')
        self._btn_rescore.setEnabled(bool(cache and p._files))
        v3.addWidget(self._btn_rescore)
        lay.addWidget(g3)

        # ── ExifTool ──────────────────────────────────────────────────
        g4 = QGroupBox('ExifTool'); v4 = QVBoxLayout(g4)
        from af_parser import EXIFTOOL_PATH as _ET
        if _ET:
            et = QLabel(f'✅ 已就绪\n路径：{_ET}')
            et.setStyleSheet('color:#6db88a;font-size:13px;')
        else:
            et = QLabel('❌ 未检测到 exiftool\nhttps://exiftool.org/')
            et.setStyleSheet('color:#c47070;font-size:13px;')
        et.setWordWrap(True)
        v4.addWidget(et)
        lay.addWidget(g4)

        # ── 确定 / 取消 ───────────────────────────────────────────────
        lay.addSpacing(4)
        row = QHBoxLayout(); row.addStretch()
        b_ok = QPushButton('确定'); b_cancel = QPushButton('取消')
        b_ok.setStyleSheet('background:#4a78b0;color:#fff;')
        b_cancel.setStyleSheet('background:#1e2029;color:#a8b0c4;')
        b_ok.clicked.connect(self.accept)
        b_cancel.clicked.connect(self.reject)
        row.addWidget(b_ok); row.addWidget(b_cancel)
        lay.addLayout(row)

    # ── 滑动条行构造助手 ──────────────────────────────────────────────

    def _make_slider_row(self, parent_layout: QVBoxLayout,
                         title: str, init_pct: int,
                         lo: int, hi: int):
        """构造「标题 + 滑动条 + 当前值标签」一行，返回 (slider, value_label)。"""
        hdr = QHBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet('color:#8898b4;font-size:13px;')
        lbl_val = QLabel(f'{init_pct / 100:.2f}')
        lbl_val.setStyleSheet(
            'color:#d0d8f0;font-size:13px;font-weight:600;'
            'min-width:36px;'
        )
        lbl_val.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if _PYSIDE6 else Qt.AlignRight | Qt.AlignVCenter
        )
        hdr.addWidget(lbl_title, 1)
        hdr.addWidget(lbl_val, 0)
        parent_layout.addLayout(hdr)

        sld = QSlider(
            Qt.Orientation.Horizontal if _PYSIDE6 else Qt.Horizontal
        )
        sld.setRange(lo, hi)
        sld.setValue(init_pct)
        sld.setTickInterval(5)
        sld.setSingleStep(1)
        sld.setPageStep(5)
        sld.setStyleSheet(_SLIDER_STYLE)

        # 实时更新数值标签
        sld.valueChanged.connect(
            lambda v, lbl=lbl_val: lbl.setText(f'{v / 100:.2f}')
        )

        parent_layout.addWidget(sld)
        return sld, lbl_val

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def confirm_delete(self):       return self._cb_confirm.isChecked()
    @property
    def lock_zoom(self):             return self._cb_lock.isChecked()
    @property
    def show_exif(self):             return self._cb_exif.isChecked()
    @property
    def burst_label_head_only(self): return self._cb_group_head.isChecked()
    @property
    def yolo_bird_conf(self):        return self._sld_bird.value() / 100.0
    @property
    def yolo_eye_conf(self):         return self._sld_eye.value()  / 100.0

