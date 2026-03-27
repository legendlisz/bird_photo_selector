#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt 兼容层：统一 PySide6 / PyQt5 的导入与常量别名。"""

import sys

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QFileDialog, QToolBar, QSizePolicy, QStatusBar,
        QMessageBox, QDialog, QGroupBox, QCheckBox, QPushButton, QMenu,
        QListWidget, QListWidgetItem, QSplitter, QScrollArea, QFrame,
        QAbstractItemView, QSlider, QToolButton
    )
    from PySide6.QtCore import Qt, QThread, Signal, QPoint, QSize, QRectF, QRect, QObject, QTimer
    from PySide6.QtGui import (
        QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon,
        QKeySequence, QAction, QPalette, QCursor, QBrush,
        QResizeEvent, QWheelEvent, QMouseEvent, QFontMetrics,
        QLinearGradient
    )
    _PYSIDE6 = True
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QFileDialog, QToolBar, QSizePolicy, QStatusBar,
            QMessageBox, QAction, QDialog, QGroupBox, QCheckBox, QPushButton, QMenu,
            QListWidget, QListWidgetItem, QSplitter, QScrollArea, QFrame,
            QAbstractItemView, QSlider, QToolButton
        )
        from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QPoint, QSize, QRectF, QRect, QObject, QTimer
        from PyQt5.QtGui import (
            QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon,
            QKeySequence, QPalette, QCursor, QBrush,
            QResizeEvent, QWheelEvent, QMouseEvent, QFontMetrics,
            QLinearGradient
        )
        _PYSIDE6 = False
    except ImportError:
        print("❌ 请安装 PySide6：pip install PySide6"); sys.exit(1)

# ── 枚举/常量兼容别名 ─────────────────────────────────────────────────
try:
    _FMT_RGB888   = QImage.Format.Format_RGB888
    _FMT_RGBA8888 = QImage.Format.Format_RGBA8888
    _ANTIALIASING = QPainter.RenderHint.Antialiasing
    _ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    _ALIGN_LEFT   = Qt.AlignmentFlag.AlignLeft
    _ALIGN_VCENTER= Qt.AlignmentFlag.AlignVCenter
    _STRONG_FOCUS = Qt.FocusPolicy.StrongFocus
    _SP_EXPAND    = QSizePolicy.Policy.Expanding
    _LMB          = Qt.MouseButton.LeftButton
    _RMB          = Qt.MouseButton.RightButton
    _KEEP_AR      = Qt.AspectRatioMode.KeepAspectRatio
    _SMOOTH_XFORM = Qt.TransformationMode.SmoothTransformation
    _ARROW_CUR    = Qt.CursorShape.ArrowCursor
    _HAND_CUR     = Qt.CursorShape.ClosedHandCursor
    _SOLID_LINE   = Qt.PenStyle.SolidLine
    _DASH_LINE    = Qt.PenStyle.DashLine
    _NO_BRUSH     = Qt.BrushStyle.NoBrush
    _K_LEFT       = Qt.Key.Key_Left;  _K_RIGHT = Qt.Key.Key_Right
    _K_ESC        = Qt.Key.Key_Escape
    _K_DEL        = Qt.Key.Key_Delete; _K_BACK = Qt.Key.Key_Backspace
    _MOD_CTRL     = Qt.KeyboardModifier.ControlModifier
    _MOD_SHIFT    = Qt.KeyboardModifier.ShiftModifier
    _TXT_SINGLE   = Qt.TextFlag.TextSingleLine
    _ROUND_CAP    = Qt.PenCapStyle.RoundCap
    _CTX_CUSTOM   = Qt.ContextMenuPolicy.CustomContextMenu
    _NO_FOCUS     = Qt.FocusPolicy.NoFocus
except AttributeError:
    _FMT_RGB888   = QImage.Format_RGB888
    _FMT_RGBA8888 = QImage.Format_RGBA8888
    _ANTIALIASING = QPainter.Antialiasing
    _ALIGN_CENTER = Qt.AlignCenter
    _ALIGN_LEFT   = Qt.AlignLeft
    _ALIGN_VCENTER= Qt.AlignVCenter
    _STRONG_FOCUS = Qt.StrongFocus
    _NO_FOCUS     = Qt.NoFocus
    _SP_EXPAND    = QSizePolicy.Expanding
    _LMB          = Qt.LeftButton;  _RMB = Qt.RightButton
    _KEEP_AR      = Qt.KeepAspectRatio
    _SMOOTH_XFORM = Qt.SmoothTransformation
    _ARROW_CUR    = Qt.ArrowCursor;  _HAND_CUR = Qt.ClosedHandCursor
    _SOLID_LINE   = Qt.SolidLine;    _DASH_LINE = Qt.DashLine
    _NO_BRUSH     = Qt.NoBrush
    _K_LEFT       = Qt.Key_Left;  _K_RIGHT = Qt.Key_Right
    _K_ESC        = Qt.Key_Escape
    _K_DEL        = Qt.Key_Delete; _K_BACK = Qt.Key_Backspace
    _MOD_CTRL     = Qt.ControlModifier;  _MOD_SHIFT = Qt.ShiftModifier
    _TXT_SINGLE   = Qt.TextSingleLine
    _ROUND_CAP    = Qt.RoundCap
    _CTX_CUSTOM   = Qt.CustomContextMenu
