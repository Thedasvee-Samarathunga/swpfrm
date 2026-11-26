"""Model registry + downloader.

Models are fetched on first use from the FaceFusion model mirrors (Hugging Face,
with GitHub releases as a fallback) and verified against FaceFusion's CRC32
.hash files when those are available.
"""
from __future__ import annotations

import os
import ssl
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _SSL = ssl.create_default_context()

ProgressFn = Callable[[str, int, int], None]  # (label, done_bytes, total_bytes)


@dataclass(frozen=True)
class ModelInfo:
    key: str
    release: str
    filename: str
    label: str


MODELS: dict[str, ModelInfo] = {m.key: m for m in [
    # analysis
    ModelInfo('retinaface_10g', 'models-3.0.0', 'retinaface_10g.onnx', 'Face detector'),
    ModelInfo('arcface_w600k_r50', 'models-3.0.0', 'arcface_w600k_r50.onnx', 'Face recogniser'),
    ModelInfo('2dfan4', 'models-3.0.0', '2dfan4.onnx', 'Landmark model'),
    # swappers
    ModelInfo('alphaface_256', 'models-3.9.0', 'alphaface_256.onnx', 'AlphaFace'),
    ModelInfo('hyperswap_1a_256', 'models-3.3.0', 'hyperswap_1a_256.onnx', 'HyperSwap 1A'),
    ModelInfo('hyperswap_1b_256', 'models-3.3.0', 'hyperswap_1b_256.onnx', 'HyperSwap 1B'),
    ModelInfo('hyperswap_1c_256', 'models-3.3.0', 'hyperswap_1c_256.onnx', 'HyperSwap 1C'),
    ModelInfo('inswapper_128_fp16', 'models-3.0.0', 'inswapper_128_fp16.onnx', 'InSwapper 128'),
    # expression restorer (LivePortrait)
    ModelInfo('live_portrait_feature_extractor', 'models-3.0.0', 'live_portrait_feature_extractor.onnx',
              'Expression model 1/3'),
    ModelInfo('live_portrait_motion_extractor', 'models-3.0.0', 'live_portrait_motion_extractor.onnx',
              'Expression model 2/3'),
    ModelInfo('live_portrait_generator', 'models-3.0.0', 'live_portrait_generator.onnx', 'Expression model 3/3'),
    # enhancers
    ModelInfo('gfpgan_1.4', 'models-3.0.0', 'gfpgan_1.4.onnx', 'GFPGAN 1.4'),
    ModelInfo('gpen_bfr_512', 'models-3.0.0', 'gpen_bfr_512.onnx', 'GPEN 512'),
    # masks
    ModelInfo('xseg_1', 'models-3.1.0', 'xseg_1.onnx', 'Occlusion mask'),
    ModelInfo('bisenet_resnet_34', 'models-3.0.0', 'bisenet_resnet_34.onnx', 'Face shape mask'),
]}

MIRRORS = [
    'https://huggingface.co/facefusion/{release}/resolve/main/{file}',
    'https://github.com/facefusion/facefusion-assets/releases/download/{release}/{file}',
]


def models_dir() -> str:
    override = os.environ.get('SWAPFRAME_MODELS')
    if override:
        path = override
    elif os.name == 'nt':
        path = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'Swapframe', 'models')
    else:
        path = os.path.join(os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share')), 'swapframe', 'models')
    os.makedirs(path, exist_ok=True)
    return path


def model_path(key: str) -> str:
    return os.path.join(models_dir(), MODELS[key].filename)


def is_downloaded(key: str) -> bool:
    return os.path.isfile(model_path(key))


def _open(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Swapframe/1.2'})
    return urllib.request.urlopen(req, timeout=timeout, context=_SSL)


def _fetch_hash(release: str, filename: str) -> Optional[str]:
    stem = os.path.splitext(filename)[0]
    for mirror in MIRRORS:
        try:
            with _open(mirror.format(release=release, file=stem + '.hash'), timeout=15) as r:
                text = r.read(64).decode('ascii', 'ignore').strip().lower()
                if len(text) == 8 and all(c in '0123456789abcdef' for c in text):
                    return text
        except Exception:
            continue
    return None


def ensure(key: str, progress: Optional[ProgressFn] = None, cancelled: Callable[[], bool] = lambda: False) -> str:
    """Return the local path of a model, downloading it if necessary."""
    info = MODELS[key]
    dest = model_path(key)
    if os.path.isfile(dest):
        return dest

    expected = _fetch_hash(info.release, info.filename)
    part = dest + '.part'
    last_error: Exception | None = None

    for mirror in MIRRORS:
        url = mirror.format(release=info.release, file=info.filename)
        try:
            with _open(url) as r, open(part, 'wb') as f:
                total = int(r.headers.get('Content-Length') or 0)
                done, crc = 0, 0
                while True:
                    if cancelled():
                        raise InterruptedError('Download cancelled')
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    crc = zlib.crc32(chunk, crc)
                    done += len(chunk)
                    if progress:
                        progress(info.label, done, total)
            if total and done != total:
                raise IOError(f'Incomplete download ({done}/{total} bytes)')
            if expected and format(crc & 0xFFFFFFFF, '08x') != expected:
                raise IOError('Checksum mismatch')
            os.replace(part, dest)
            return dest
        except InterruptedError:
            _silent_remove(part)
            raise
        except Exception as e:  # try next mirror
            last_error = e
            _silent_remove(part)
    raise RuntimeError(f'Could not download {info.label}: {last_error}')


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
