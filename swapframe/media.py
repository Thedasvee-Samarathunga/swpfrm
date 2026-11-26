"""Image + video I/O (unicode-safe, ffmpeg based for video)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.gif'}

_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXT


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def read_image(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def write_image(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower() or '.png'
    params = [cv2.IMWRITE_JPEG_QUALITY, 96] if ext in ('.jpg', '.jpeg') else []
    if ext == '.webp':
        params = [cv2.IMWRITE_WEBP_QUALITY, 96]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError('Could not encode image')
    buf.tofile(path)


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float

    @property
    def frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))


def _run(args, **kw):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW, **kw)


def grab_frame(path: str, seconds: float = 0.0) -> Optional[np.ndarray]:
    args = [ffmpeg_exe(), '-v', 'error', '-ss', f'{max(0.0, seconds):.3f}', '-i', path,
            '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-']
    res = _run(args)
    if not res.stdout:
        return None
    return cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)


def probe(path: str) -> Optional[VideoInfo]:
    res = _run([ffmpeg_exe(), '-hide_banner', '-i', path])
    text = res.stderr.decode('utf-8', 'ignore')
    fps = 30.0
    m = re.search(r'Video:.*?(\d+(?:\.\d+)?) fps', text)
    if m:
        fps = float(m.group(1))
    else:
        m = re.search(r'Video:.*?(\d+(?:\.\d+)?) tbr', text)
        if m:
            fps = float(m.group(1))
    duration = 0.0
    m = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    first = grab_frame(path, 0)
    if first is None:
        return None
    h, w = first.shape[:2]  # post-rotation size
    return VideoInfo(w, h, fps if fps > 0 else 30.0, duration)


def iter_frames(path: str, info: VideoInfo) -> Iterator[np.ndarray]:
    args = [ffmpeg_exe(), '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{info.width}x{info.height}', '-']
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
                            bufsize=info.width * info.height * 3 * 4)
    size = info.width * info.height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


class VideoWriter:
    """Encodes BGR frames to H.264 and copies the source audio track (if any)."""

    def __init__(self, out_path: str, source_path: str, info: VideoInfo, crf: int = 16):
        w, h = info.width - info.width % 2, info.height - info.height % 2
        self.size = (w, h)
        args = [ffmpeg_exe(), '-y', '-v', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{info.fps:.5f}', '-i', '-',
                '-i', source_path,
                '-map', '0:v:0', '-map', '1:a:0?',
                '-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-shortest',
                '-metadata', 'comment=Contains AI face swap (Swapframe)',
                '-movflags', '+faststart', out_path]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = frame[:h, :w]
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode('utf-8', 'ignore')
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f'ffmpeg failed: {err.strip()[-400:]}')

    def abort(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()
        self.proc.wait()
