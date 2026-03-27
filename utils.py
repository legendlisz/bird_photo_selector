import os
import re
import sys
import shutil
import json
import csv
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

try:
    from PySide6.QtGui import QImage, QPixmap, QColor
    _PYSIDE6 = True
except ImportError:
    from PyQt5.QtGui import QImage, QPixmap, QColor
    _PYSIDE6 = False

try:
    _FMT_RGB888   = QImage.Format.Format_RGB888
    _FMT_RGBA8888 = QImage.Format.Format_RGBA8888
except AttributeError:
    _FMT_RGB888   = QImage.Format_RGB888
    _FMT_RGBA8888 = QImage.Format_RGBA8888

from af_parser import AFPoint

# ── 常量 ──
FLAG_NONE = 0
FLAG_PICK = 1
FLAG_REJECT = -1

CSV_FILENAME = '.raw_scores.csv'
FLAG_FILENAME = '.raw_flags.json'

SHARP_HI = 300
SHARP_LO = 80
CLR_SHARP  = QColor(109, 184, 138)
CLR_MEDIUM = QColor(201, 165,  90)
CLR_SOFT   = QColor(196, 112, 112)

# ── 辅助函数 ──
def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def is_scan_noise_file(name: str) -> bool:
    n = str(name)
    return n.startswith('._') or n.startswith('.')

def delete_hint_short_text(has_send2trash: bool) -> str:
    return '文件将移入回收站' if has_send2trash else '⚠ 未安装 send2trash，删除功能已禁用'

def delete_hint_text(has_send2trash: bool) -> str:
    return '文件将移入回收站，可恢复。' if has_send2trash else '⚠ 未安装 send2trash，删除功能已禁用。'

def move_to_trash_strict(path: str, has_send2trash: bool, send2trash_fn):
    if not has_send2trash or send2trash_fn is None:
        raise RuntimeError('send2trash 未安装，删除已被阻止')
    ap = str(Path(path).resolve())
    if sys.platform == 'darwin':
        script = 'tell application "Finder" to delete POSIX file ' + json.dumps(ap)
        subprocess.run(['osascript', '-e', script], check=True,
                       creationflags=0x08000000 if sys.platform=='win32' else 0)
        return
    send2trash_fn(ap)

def find_exiftool():
    candidate_dirs = []
    if getattr(sys, 'frozen', False):
        candidate_dirs.append(os.path.dirname(sys.executable))
    if hasattr(sys, '_MEIPASS'):
        candidate_dirs.append(sys._MEIPASS)
    candidate_dirs.append(os.path.dirname(os.path.abspath(__file__)))

    seen = set()
    preferred = []
    fallback = []
    for d in candidate_dirs:
        if not d or d in seen:
            continue
        seen.add(d)
        has_runtime = os.path.isdir(os.path.join(d, 'exiftool_files'))
        for n in ('exiftool.exe', 'exiftool'):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                if has_runtime:
                    preferred.append(p)
                else:
                    fallback.append(p)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]

    for n in ('exiftool', 'exiftool.exe'):
        p = shutil.which(n)
        if p: return p
    for p in (r'C:\Windows\exiftool.exe',
              r'C:\Program Files\exiftool\exiftool.exe',
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exiftool.exe')):
        if os.path.isfile(p): return p
    for p in ('/usr/local/bin/exiftool', '/opt/homebrew/bin/exiftool'):
        if os.path.isfile(p): return p
    return None

def _load_prefs(prefs_path: Path):
    try:
        if prefs_path.exists():
            return json.loads(prefs_path.read_text(encoding='utf-8'))
    except Exception as e: print(f'[prefs] {e}')
    return {}

def _save_prefs(prefs_path: Path, d):
    try: prefs_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e: print(f'[prefs] {e}')

def ndarray_to_qimage(arr) -> Optional[QImage]:
    if arr.ndim != 3: return None
    h, w, c = arr.shape
    fmt = _FMT_RGB888 if c == 3 else (_FMT_RGBA8888 if c == 4 else None)
    if fmt is None: return None
    return QImage(arr.tobytes(), w, h, w * c, fmt).copy()

def _qimage_to_gray(qi: QImage) -> Optional[np.ndarray]:
    if qi is None or qi.isNull(): return None
    if qi.format() != _FMT_RGB888: qi = qi.convertToFormat(_FMT_RGB888)
    W, H = qi.width(), qi.height()
    if W <= 0 or H <= 0: return None
    bpl = int(qi.bytesPerLine())
    if bpl <= 0: return None
    try:
        ptr = qi.constBits()
        if hasattr(ptr, 'setsize'): ptr.setsize(H * bpl)
        buf = np.frombuffer(ptr, dtype=np.uint8, count=H * bpl)
        rows = buf.reshape(H, bpl)
        rgb = rows[:, :W * 3].reshape(H, W, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"[gray] {e}"); return None

def _qimage_to_bgr(qi: QImage) -> Optional[np.ndarray]:
    if qi is None or qi.isNull(): return None
    if qi.format() != _FMT_RGB888: qi = qi.convertToFormat(_FMT_RGB888)
    W, H = qi.width(), qi.height()
    if W <= 0 or H <= 0: return None
    bpl = int(qi.bytesPerLine())
    if bpl <= 0: return None
    try:
        ptr = qi.constBits()
        if hasattr(ptr, 'setsize'): ptr.setsize(H * bpl)
        buf = np.frombuffer(ptr, dtype=np.uint8, count=H * bpl)
        rows = buf.reshape(H, bpl)
        rgb = rows[:, :W * 3].reshape(H, W, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[bgr] {e}"); return None

def _pixmap_to_gray(pm: QPixmap) -> Optional[np.ndarray]:
    return _qimage_to_gray(pm.toImage()) if pm is not None else None

def calculate_laplacian_variance(img, x, y, w, h):
    try:
        x = max(0, min(x, img.shape[1] - 1))
        y = max(0, min(y, img.shape[0] - 1))
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        if w <= 0 or h <= 0: return 0.0
        return cv2.Laplacian(img[y:y+h, x:x+w], cv2.CV_64F).var()
    except Exception as e:
        print(f"[lap] {e}"); return 0.0

def compute_center_sharpness_qimage(qi: QImage) -> float:
    gray = _qimage_to_gray(qi)
    if gray is None: return 0.0
    H, W = gray.shape
    return calculate_laplacian_variance(gray, W // 3, H // 3, W // 3, H // 3)

def _sharpness_color(v: float) -> QColor:
    return CLR_SHARP if v >= SHARP_HI else (CLR_MEDIUM if v >= SHARP_LO else CLR_SOFT)

def calculate_image_sharpness(gray: np.ndarray, af_points: List[AFPoint], ref_w: int, ref_h: int, pm_w: int, pm_h: int) -> float:
    if not af_points:
        H, W = gray.shape
        return calculate_laplacian_variance(gray, W // 3, H // 3, W // 3, H // 3)
    if not ref_w or not ref_h:
        H, W = gray.shape
        return calculate_laplacian_variance(gray, W // 3, H // 3, W // 3, H // 3)
    vals = []
    sx, sy = pm_w / ref_w, pm_h / ref_h
    for pt in af_points:
        x = int(pt.cx * sx - pt.w * sx / 2)
        y = int(pt.cy * sy - pt.h * sy / 2)
        w = int(pt.w * sx)
        h = int(pt.h * sy)
        v = calculate_laplacian_variance(gray, x, y, w, h)
        pt.sharpness = v
        vals.append(v)
    return sum(vals) / len(vals) if vals else 0.0


def _xmp_path(raw_path: str) -> str:
    return str(Path(raw_path).with_suffix('.xmp'))


def is_our_xmp(xmp_path: str) -> bool:
    if not os.path.isfile(xmp_path):
        return False
    try:
        with open(xmp_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1024)
            return 'RAW Photo Viewer' in content
    except Exception:
        return False


def write_xmp_flag(raw_path: str, flag: int, exiftool_path: Optional[str] = None, app_version: str = ''):
    xmp_file = _xmp_path(raw_path)
    base = os.path.basename(raw_path)

    if exiftool_path is None:
        exiftool_path = find_exiftool()

    if flag != FLAG_PICK:
        if os.path.exists(xmp_file) and is_our_xmp(xmp_file):
            try:
                os.remove(xmp_file)
                return
            except Exception:
                pass

        if not os.path.exists(xmp_file):
            return

        if exiftool_path:
            try:
                cmd = [exiftool_path, '-overwrite_original', '-q', '-q']
                cmd.append('-XMP:Rating=0')
                cmd.append('-XMP:Label=')
                cmd.append('-XMP:Subject-=Pick')
                cmd.append('-XMP:Subject-=Reject')
                cmd.append(xmp_file)
                _et_dir = os.path.dirname(os.path.abspath(exiftool_path)) if exiftool_path else None
                _env = os.environ.copy()
                if _et_dir:
                    _env['PATH'] = _et_dir + os.pathsep + _env.get('PATH', '')
                    _env['PERL5LIB'] = os.path.join(_et_dir, 'exiftool_files', 'lib')
                subprocess.run(
                    cmd,
                    check=True,
                    creationflags=0x08000000 if sys.platform=='win32' else 0,
                    cwd=_et_dir,
                    env=_env
                )
            except Exception as e:
                print(f"[XMP] ExifTool 清除标记失败 {base}: {e}")
        return

    rating, label, sub = "1", "Green", "Pick"

    if exiftool_path:
        try:
            cmd = [exiftool_path, '-overwrite_original', '-q', '-q']
            cmd.append(f'-XMP:Rating={rating}')
            cmd.append(f'-XMP:Label={label}')
            cmd.append('-XMP:Subject-=Pick')
            cmd.append('-XMP:Subject-=Reject')
            if sub:
                cmd.append(f'-XMP:Subject+={sub}')
            cmd.append(xmp_file)
            _et_dir = os.path.dirname(os.path.abspath(exiftool_path)) if exiftool_path else None
            _env = os.environ.copy()
            if _et_dir:
                _env['PATH'] = _et_dir + os.pathsep + _env.get('PATH', '')
                _env['PERL5LIB'] = os.path.join(_et_dir, 'exiftool_files', 'lib')
            subprocess.run(
                cmd,
                check=True,
                creationflags=0x08000000 if sys.platform=='win32' else 0,
                cwd=_et_dir,
                env=_env
            )
            return
        except Exception as e:
            print(f"[XMP] ExifTool 更新失败 {base}: {e}")
            if os.path.exists(xmp_file) and not is_our_xmp(xmp_file):
                print(f"[XMP] 放弃覆盖非本程序创建的 XMP: {base}")
                return

    if os.path.exists(xmp_file) and not is_our_xmp(xmp_file):
        print(f"[XMP] 警告: 未安装 ExifTool，无法安全更新非本程序创建的 XMP: {base}")
        return

    subject_block = (
        f'   <lr:hierarchicalSubject>\n    <rdf:Bag>\n     <rdf:li>{sub}</rdf:li>\n    </rdf:Bag>\n   </lr:hierarchicalSubject>\n'
        if sub else ''
    )
    label_line = f'   xmp:Label="{label}"\n' if label else ''
    ver = app_version or ''

    xmp_content = textwrap.dedent(f"""\
<?xpacket begin='\xef\xbb\xbf' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk='RAW Photo Viewer v{ver}'>
 <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
  <rdf:Description rdf:about=''
   xmlns:xmp='http://ns.adobe.com/xap/1.0/'
   xmlns:lr='http://ns.adobe.com/lightroom/1.0/'
   xmp:Rating="{rating}"
{label_line}  >
{subject_block}  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
""")

    try:
        Path(xmp_file).write_text(xmp_content, encoding='utf-8')
    except Exception as e:
        print(f"[XMP] 写入失败 {base}: {e}")


class FlagCache:
    def __init__(self, folder: str):
        self._folder = folder
        self._json = os.path.join(folder, FLAG_FILENAME)
        self._data: Dict[str, int] = {}
        self._dirty = False

    def load(self):
        self._data = {}
        if not os.path.isfile(self._json):
            return
        try:
            raw = json.loads(Path(self._json).read_text(encoding='utf-8'))
            for k, v in raw.items():
                if v in (FLAG_PICK, FLAG_REJECT, FLAG_NONE):
                    self._data[k] = v
        except Exception as e:
            print(f"[FlagCache] 读取失败: {e}")

    def get(self, path: str) -> int:
        return self._data.get(os.path.basename(path), FLAG_NONE)

    def set(self, path: str, flag: int):
        key = os.path.basename(path)
        if flag == FLAG_NONE:
            self._data.pop(key, None)
        else:
            self._data[key] = flag
        self._dirty = True

    def remove(self, path: str):
        key = os.path.basename(path)
        if key in self._data:
            del self._data[key]
            self._dirty = True

    def all_with_flag(self, paths: List[str], flag: int) -> List[str]:
        return [p for p in paths if self.get(p) == flag]

    def save(self):
        try:
            Path(self._json).write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding='utf-8')
            self._dirty = False
        except Exception as e:
            print(f"[FlagCache] 写入失败: {e}")

    def save_if_dirty(self):
        if self._dirty:
            self.save()

    def count_pick(self) -> int:
        return sum(1 for v in self._data.values() if v == FLAG_PICK)

    def count_reject(self) -> int:
        return sum(1 for v in self._data.values() if v == FLAG_REJECT)


class ScoreCache:
    FIELDS = ['filename', 'sharpness', 'scored_at', 'fused_score', 'fused_scored_at', 'bird_box']

    def __init__(self, folder: str):
        self._folder = folder
        self._csv = os.path.join(folder, CSV_FILENAME)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._dirty: bool = False

    def load(self) -> bool:
        self._data = {}
        if not os.path.isfile(self._csv):
            return False
        try:
            with open(self._csv, newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    fn = row.get('filename', '').strip()
                    if not fn:
                        continue
                    try:
                        sh = float(row.get('sharpness', 0) or 0)
                    except Exception:
                        sh = 0.0
                    fs_raw = row.get('fused_score', '')
                    try:
                        fused = float(fs_raw) if str(fs_raw).strip() else None
                    except Exception:
                        fused = None
                    self._data[fn] = {
                        'sharpness': sh,
                        'scored_at': row.get('scored_at', ''),
                        'fused_score': fused,
                        'fused_scored_at': row.get('fused_scored_at', ''),
                        'bird_box': row.get('bird_box', ''),
                    }
            return bool(self._data)
        except Exception as e:
            print(f"[ScoreCache] {e}")
            return False

    def get(self, filename: str) -> Optional[float]:
        rec = self._data.get(os.path.basename(filename))
        return rec['sharpness'] if rec else None

    def get_fused(self, filename: str) -> Optional[float]:
        rec = self._data.get(os.path.basename(filename))
        v = rec.get('fused_score') if rec else None
        return float(v) if v is not None else None

    def get_bird_box(self, filename: str) -> str:
        rec = self._data.get(os.path.basename(filename))
        return rec.get('bird_box', '') if rec else ''

    def has(self, filename: str) -> bool:
        return os.path.basename(filename) in self._data

    def missing(self, paths: List[str]) -> List[str]:
        return [p for p in paths if not self.has(p)]

    def set(self, filename: str, sharpness: float, save_now: bool = False):
        key = os.path.basename(filename)
        rec = self._data.get(key) or {}
        rec['sharpness'] = round(sharpness, 2)
        rec['scored_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        if 'fused_score' not in rec:
            rec['fused_score'] = None
        if 'fused_scored_at' not in rec:
            rec['fused_scored_at'] = ''
        if 'bird_box' not in rec:
            rec['bird_box'] = ''
        self._data[key] = rec
        self._dirty = True
        if save_now:
            self.save()

    def has_fused(self, filename: str) -> bool:
        key = os.path.basename(filename)
        rec = self._data.get(key)
        return rec is not None and rec.get('fused_score') is not None

    def missing_fused(self, paths: List[str]) -> List[str]:
        return [p for p in paths if not self.has_fused(p)]

    def set_fused(self, filename: str, fused_score: float, bird_box: str = '', save_now: bool = False):
        key = os.path.basename(filename)
        rec = self._data.get(key) or {}
        try:
            rec['fused_score'] = round(float(fused_score), 2)
        except Exception:
            rec['fused_score'] = None
        rec['fused_scored_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        rec['bird_box'] = bird_box
        if 'sharpness' not in rec:
            rec['sharpness'] = 0.0
        if 'scored_at' not in rec:
            rec['scored_at'] = ''
        self._data[key] = rec
        self._dirty = True
        if save_now:
            self.save()

    def remove(self, filename: str):
        key = os.path.basename(filename)
        if key in self._data:
            del self._data[key]
            self._dirty = True

    def retain(self, paths: List[str]) -> int:
        keep = {os.path.basename(p) for p in paths}
        stale = [k for k in self._data.keys() if k not in keep]
        for k in stale:
            del self._data[k]
        if stale:
            self._dirty = True
        return len(stale)

    def save(self):
        rows = []
        for k, v in self._data.items():
            rec: Dict[str, Any] = {'filename': k}
            rec.update(v or {})
            if rec.get('fused_score') is None:
                rec['fused_score'] = ''
            if rec.get('fused_scored_at') is None:
                rec['fused_scored_at'] = ''
            if rec.get('bird_box') is None:
                rec['bird_box'] = ''
            rows.append(rec)
        rows.sort(key=lambda r: natural_key(r['filename']))
        try:
            with open(self._csv, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                w.writeheader()
                w.writerows(rows)
            self._dirty = False
        except Exception as e:
            print(f"[ScoreCache] 写入失败: {e}")

    def save_if_dirty(self):
        if self._dirty:
            self.save()

    def count(self) -> int:
        return len(self._data)

    @property
    def csv_path(self) -> str:
        return self._csv