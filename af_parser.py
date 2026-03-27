import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

CREATE_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0

SUPPORTED_EXTS = {'.nef', '.cr2', '.cr3', '.arw', '.orf'}
BRAND_BY_EXT   = {'.nef':'Nikon', '.cr2':'Canon', '.cr3':'Canon',
                  '.arw':'Sony',  '.orf':'OM'}

EXIFTOOL_PATH = None

def set_exiftool_path(path: Optional[str]):
    global EXIFTOOL_PATH
    EXIFTOOL_PATH = path

def _first_int(d: dict, *keys) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try: return int(float(str(v)))
            except: pass
    return None

def _first_str(d: dict, *keys) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            s = str(v).strip()
            if s and s.lower() not in ('n/a','none',''): return s
    return None

def _fmt_focus_dist(raw: Optional[str]) -> Optional[str]:
    if not raw: return None
    s = raw.strip().lower()
    if s in ('inf','infinity','∞','65535','65534','0xffff'): return '无穷远'
    try:
        val = float(s)
        if val<=0: return None
        if val>=1e4: return '无穷远'
        return f'{val:.1f} 米'
    except: return None

@dataclass
class AFPoint:
    cx: float; cy: float; w: float; h: float
    in_focus: bool = True; selected: bool = True; sharpness: float = 0.0

class AFExtractor:
    @staticmethod
    def _parse_floats(s: Optional[str]) -> List[float]:
        if not s: return []
        try: return [float(v) for v in s.split()]
        except: return []

    @staticmethod
    def extract(path):
        if not EXIFTOOL_PATH: return [],"exiftool 未安装",{}
        exif = AFExtractor._run(path)
        if not exif: return [],"exiftool 无数据",{}
        pts, msg = AFExtractor._parse(path, exif)
        return pts, msg, exif

    @staticmethod
    def _parse(path, exif):
        """从已有的 exif dict 解析 AF 点，不调用 exiftool。"""
        ext = Path(path).suffix.lower(); brand = BRAND_BY_EXT.get(ext, '')
        try:
            if brand=='Sony':    return AFExtractor._sony(exif)
            elif brand=='Nikon': return AFExtractor._nikon(exif)
            elif brand=='Canon': return AFExtractor._canon(exif)
            elif brand=='OM':    return AFExtractor._om(exif)
            else:                return [], f"未知品牌({ext})"
        except Exception as e:
            return [], f"解析异常: {e}"

    @staticmethod
    def extract_batch(paths: List[str]) -> Dict[str, dict]:
        """
        一次 exiftool 调用处理多个文件（每批 50 张），
        返回 {归一化路径: exif_dict}。
        比逐张调用快约 50 倍（减少进程启动开销）。
        """
        if not EXIFTOOL_PATH or not paths: return {}
        result: Dict[str, dict] = {}
        CHUNK = 50
        for i in range(0, len(paths), CHUNK):
            chunk = paths[i:i+CHUNK]
            try:
                _et_dir = os.path.dirname(os.path.abspath(EXIFTOOL_PATH))
                _env = os.environ.copy()
                _env['PATH'] = _et_dir + os.pathsep + _env.get('PATH', '')
                _env['PERL5LIB'] = os.path.join(_et_dir, 'exiftool_files', 'lib')
                r = subprocess.run(
                    [EXIFTOOL_PATH, '-j', '-a', '-u', '-G1', '-n'] + chunk,
                    capture_output=True, text=True, timeout=120,
                    encoding='utf-8', errors='replace',
                    creationflags=CREATE_NO_WINDOW,
                    cwd=_et_dir,
                    env=_env
                )
                if r.returncode == 0 and r.stdout.strip():
                    data = json.loads(r.stdout)
                    if isinstance(data, list):
                        for item in data:
                            src = str(Path(item.get('SourceFile', '')).resolve())
                            result[src] = item
            except subprocess.TimeoutExpired:
                print(f"[exiftool batch] 超时（批次 {i//CHUNK+1}）")
            except Exception as e:
                print(f"[exiftool batch] {e}")
        return result

    @staticmethod
    def _run(path):
        """单文件调用 exiftool。"""
        try:
            _et_dir = os.path.dirname(os.path.abspath(EXIFTOOL_PATH))
            _env = os.environ.copy()
            _env['PATH'] = _et_dir + os.pathsep + _env.get('PATH', '')
            _env['PERL5LIB'] = os.path.join(_et_dir, 'exiftool_files', 'lib')
            r = subprocess.run(
                [EXIFTOOL_PATH, '-j', '-a', '-u', '-G1', '-n', path],
                capture_output=True, text=True, timeout=20,
                encoding='utf-8', errors='replace',
                creationflags=CREATE_NO_WINDOW,
                cwd=_et_dir,
                env=_env
            )
            if r.returncode == 0 and r.stdout.strip():
                d = json.loads(r.stdout)
                if isinstance(d, list) and d:
                    return d[0]
        except subprocess.TimeoutExpired:
            print("[exiftool] 超时")
        except Exception as e:
            print(f"[exiftool] {e}")
        return {}

    @staticmethod
    def _run_focus_mode_text(path: str) -> str:
        if not EXIFTOOL_PATH: return ''
        try:
            _et_dir = os.path.dirname(os.path.abspath(EXIFTOOL_PATH))
            _env = os.environ.copy()
            _env['PATH'] = _et_dir + os.pathsep + _env.get('PATH', '')
            _env['PERL5LIB'] = os.path.join(_et_dir, 'exiftool_files', 'lib')
            r = subprocess.run(
                [EXIFTOOL_PATH, '-j', '-a', '-u', '-G1', '-q', '-q',
                 '-FocusMode', '-AFMode', '-AFAreaMode', '-AFDrive',
                 '-FocusSetting', '-FocusType', '-AFPointMode', path],
                capture_output=True, text=True, timeout=6,
                encoding='utf-8', errors='replace',
                creationflags=CREATE_NO_WINDOW,
                cwd=_et_dir,
                env=_env
            )
            if r.returncode == 0 and r.stdout.strip():
                d = json.loads(r.stdout)
                if isinstance(d, list) and d and isinstance(d[0], dict):
                    item = d[0]
                    fm = _first_str(item,
                        'Composite:FocusMode',
                        'MakerNotes:FocusMode','Nikon:FocusMode','Canon:FocusMode','Sony:FocusMode','Olympus:FocusMode',
                        'ExifIFD:FocusMode','EXIF:FocusMode','FocusMode')
                    if not fm:
                        fm = _first_str(item,
                            'Composite:AFMode',
                            'MakerNotes:AFMode','Nikon:AFMode','Canon:AFMode','Sony:AFMode','Olympus:AFMode','AFMode')
                    if not fm:
                        fm = _first_str(item,
                            'Composite:AFAreaMode',
                            'MakerNotes:AFAreaMode','Nikon:AFAreaMode','Canon:AFAreaMode','Sony:AFAreaMode','Olympus:AFAreaMode','AFAreaMode')
                    return (fm or '').strip()
        except subprocess.TimeoutExpired:
            return ''
        except Exception:
            return ''
        return ''

    @staticmethod
    def _sony(exif):
        loc = AFExtractor._parse_floats(_first_str(exif,'Sony:FocusLocation','MakerNotes:FocusLocation','FocusLocation'))
        if not loc or len(loc)<4: return [],"Sony: 未找到有效的 FocusLocation"
        try:
            cx,cy,w,h=loc[2],loc[3],80.0,80.0
            fs = AFExtractor._parse_floats(_first_str(exif,'Sony:FocusFrameSize','FocusFrameSize'))
            if len(fs)>=2: w,h=fs[0],fs[1]
            d=_fmt_focus_dist(_first_str(exif,'Sony:Focus Distance 2','FocusDistance2','Composite:FocusDistance2'))
            return ([AFPoint(cx=cx,cy=cy,w=w,h=h)],
                    f"Sony: 中心=({int(cx)},{int(cy)}) 框={int(w)}×{int(h)}"+(f" | 距离: {d}" if d else ""))
        except Exception as e: return [],f"Sony: {e}"

    @staticmethod
    def _nikon(exif):
        xs = AFExtractor._parse_floats(_first_str(exif,'Nikon:AFAreaXPosition','MakerNotes:AFAreaXPosition'))
        ys = AFExtractor._parse_floats(_first_str(exif,'Nikon:AFAreaYPosition','MakerNotes:AFAreaYPosition'))
        if xs and ys:
            try:
                ws = AFExtractor._parse_floats(_first_str(exif,'Nikon:AFAreaWidth','MakerNotes:AFAreaWidth')) or [80.0]*len(xs)
                hs = AFExtractor._parse_floats(_first_str(exif,'Nikon:AFAreaHeight','MakerNotes:AFAreaHeight')) or [80.0]*len(ys)
                n=min(len(xs),len(ys),len(ws),len(hs))
                pts=[AFPoint(cx=xs[i],cy=ys[i],w=max(ws[i],20),h=max(hs[i],20))
                     for i in range(n) if not(xs[i]==0 and ys[i]==0 and ws[i]==0)]
                if pts:
                    d=_fmt_focus_dist(_first_str(exif,'Nikon:FocusDistance','FocusDistance','Composite:FocusDistance'))
                    return pts,f"Nikon: {len(pts)} 个 AF 点"+(f" | 距离: {d}" if d else "")
            except Exception as e: print(f"[nikon] {e}")
        name=_first_str(exif,'Nikon:AFPointsInFocus','MakerNotes:AFPointsInFocus','Nikon:AFPoint')
        if name: return [],f"Nikon: AF点='{name}'（旧机型无坐标）"
        return [],"Nikon: 未找到对焦点数据"

    @staticmethod
    def _canon(exif):
        xs = AFExtractor._parse_floats(_first_str(exif,'Canon:AFAreaXPositions','MakerNotes:AFAreaXPositions'))
        ys = AFExtractor._parse_floats(_first_str(exif,'Canon:AFAreaYPositions','MakerNotes:AFAreaYPositions'))
        if xs and ys:
            try:
                ws = AFExtractor._parse_floats(_first_str(exif,'Canon:AFAreaWidths','MakerNotes:AFAreaWidths')) or [80.0]*len(xs)
                hs = AFExtractor._parse_floats(_first_str(exif,'Canon:AFAreaHeights','MakerNotes:AFAreaHeights')) or [80.0]*len(ys)
                iw=_first_int(exif,'EXIF:ExifImageWidth','ExifIFD:ExifImageWidth','File:ImageWidth')
                ih=_first_int(exif,'EXIF:ExifImageHeight','ExifIFD:ExifImageHeight','File:ImageHeight')
                if iw and ih and xs and (xs[0]<0 or ys[0]<0):
                    xs=[x+iw/2 for x in xs]; ys=[y+ih/2 for y in ys]
                n=min(len(xs),len(ys),len(ws),len(hs))
                pts=[AFPoint(cx=xs[i],cy=ys[i],w=max(ws[i],20),h=max(hs[i],20)) for i in range(n)]
                if pts:
                    d=_fmt_focus_dist(_first_str(exif,'Canon:FocusDistance','Composite:FocusDistance'))
                    return pts,f"Canon: {len(pts)} 个 AF 点"+(f" | 距离: {d}" if d else "")
            except Exception as e: print(f"[canon] {e}")
        idx=_first_str(exif,'Canon:AFPointsInFocus','MakerNotes:AFPointsInFocus')
        if idx: return [],f"Canon: AF点索引='{idx}'（旧机型无坐标）"
        return [],"Canon: 未找到对焦点数据"

    @staticmethod
    def _om(exif):
        fw, fh = 640.0, 480.0
        fv = AFExtractor._parse_floats(_first_str(exif, 'Olympus:AFFrameSize'))
        if len(fv) >= 2 and fv[0] > 0 and fv[1] > 0:
            fw, fh = fv[0], fv[1]
        
        nv = AFExtractor._parse_floats(_first_str(exif, 'Olympus:AFPointSelected'))
        if not nv:
            return [], "OM System: 未找到 AFPointSelected 字段"
        if len(nv) < 2 or not (0 < nv[0] < 1) or not (0 < nv[1] < 1):
            return [], f"OM System: AFPointSelected 值异常"
        cx_n, cy_n = nv[0], nv[1]
        
        w_n, h_n = 0.06, 0.06
        av = AFExtractor._parse_floats(_first_str(exif, 'Olympus:AFFocusArea', 'Olympus:AFSelectedArea'))
        if len(av) >= 4 and av[2] > 0 and av[3] > 0:
            w_n = av[2] / fw; h_n = av[3] / fh
        
        # 坐标已归一化到 [0, 1]，显式标记参考尺寸为 1×1，
        # 避免修改 exif dict 造成隐式副作用。
        exif['_OM_ref_w'] = 1
        exif['_OM_ref_h'] = 1
        pt = AFPoint(cx=cx_n, cy=cy_n, w=w_n, h=h_n, in_focus=True, selected=True)
        d = _fmt_focus_dist(_first_str(exif, 'Olympus:FocusDistance'))
        msg = (f"OM System: AF=({cx_n:.4f}, {cy_n:.4f})  框={w_n:.3f}×{h_n:.3f}"
               + (f"  |  距离: {d}" if d else ""))
        return [pt], msg