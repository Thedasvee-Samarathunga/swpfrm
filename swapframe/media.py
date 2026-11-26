"""Image + video I/O (unicode-safe, ffmpeg based for video and GIF)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.gif'}

MP4_SAFE_AUDIO = {'aac', 'mp3'}
_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXT


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def is_gif(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == '.gif'


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
    has_audio: bool = False
    frame_count: int = 0
    audio_codec: str = ''

    @property
    def frames(self) -> int:
        if self.frame_count > 0:
            return self.frame_count
        return max(1, int(round(self.duration * self.fps)))


def _run(args, **kw):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW, **kw)


def grab_frame(path: str, seconds: float = 0.0) -> Optional[np.ndarray]:
    args = [ffmpeg_exe(), '-v', 'error', '-ss', f'{max(0.0, seconds):.3f}', '-i', path,
            '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-']
    res = _run(args)
    if not res.stdout and seconds > 0:  # seeking past the end of short GIFs -> last good frame
        return grab_frame(path, 0.0)
    if not res.stdout:
        return None
    return cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)


def _count_frames(path: str) -> int:
    """Exact frame count by decoding (used for GIFs / files without a usable duration)."""
    res = _run([ffmpeg_exe(), '-v', 'error', '-i', path, '-map', '0:v:0', '-f', 'null', '-progress', '-', '-nostats'])
    found = re.findall(r'frame=(\d+)', res.stdout.decode('utf-8', 'ignore'))
    return int(found[-1]) if found else 0


def probe(path: str) -> Optional[VideoInfo]:
    res = _run([ffmpeg_exe(), '-hide_banner', '-i', path])
    text = res.stderr.decode('utf-8', 'ignore')
    fps = 0.0
    m = re.search(r'Video:.*?(\d+(?:\.\d+)?) fps', text)
    if m:
        fps = float(m.group(1))
    if fps <= 0:
        m = re.search(r'Video:.*?(\d+(?:\.\d+)?) tbr', text)
        if m:
            fps = float(m.group(1))
    if fps <= 0 or fps > 240:
        fps = 25.0 if is_gif(path) else 30.0
    duration = 0.0
    m = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    audio = re.search(r'Stream #\d+:\d+.*?: Audio: (\w+)', text)
    has_audio = audio is not None
    first = grab_frame(path, 0)
    if first is None:
        return None
    h, w = first.shape[:2]  # post-rotation size
    count = _count_frames(path) if (is_gif(path) or duration <= 0) else 0
    if count and duration <= 0:
        duration = count / fps
    return VideoInfo(w, h, fps, duration, has_audio, count, audio.group(1).lower() if audio else '')


def iter_frames(path: str, info: VideoInfo) -> Iterator[np.ndarray]:
    # constant frame rate output keeps the picture in sync with the original audio track
    args = [ffmpeg_exe(), '-v', 'error', '-i', path, '-map', '0:v:0', '-r', f'{info.fps:.5f}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{info.width}x{info.height}', '-']
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
    """Writes BGR frames, then finishes the file:

    * ``.mp4`` – H.264 video, then the original audio is muxed back in a second pass
      (copied losslessly when possible, re-encoded to AAC otherwise).
    * ``.gif`` – lossless intermediate, then a two-pass palette GIF that loops forever.

    ``close()`` returns ``'audio'``, ``'no-audio'`` (source had none) or ``'audio-failed'``.
    """

    def __init__(self, out_path: str, source_path: str, info: VideoInfo, crf: int = 16):
        self.out_path, self.source_path, self.info = out_path, source_path, info
        self.gif = is_gif(out_path)
        base = os.path.splitext(out_path)[0]
        if self.gif:
            w, h = info.width, info.height
            self.tmp = base + '.swapframe-tmp.mkv'
            codec = ['-c:v', 'ffv1', '-pix_fmt', 'bgr0']
        else:
            w, h = info.width - info.width % 2, info.height - info.height % 2
            self.tmp = base + '.swapframe-tmp.mp4'
            codec = ['-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium', '-pix_fmt', 'yuv420p']
        self.size = (w, h)
        args = [ffmpeg_exe(), '-y', '-v', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{info.fps:.5f}', '-i', '-',
                *codec, '-an', self.tmp]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = frame[:h, :w]
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> str:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode('utf-8', 'ignore')
        if self.proc.wait() != 0:
            _remove(self.tmp)
            raise RuntimeError(f'ffmpeg failed: {err.strip()[-400:]}')
        try:
            return self._finish_gif() if self.gif else self._finish_mp4()
        finally:
            _remove(self.tmp)

    def _finish_gif(self) -> str:
        fps = min(self.info.fps, 50.0)
        graph = (f'fps={fps:.3f},split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];'
                 f'[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle')
        res = _run([ffmpeg_exe(), '-y', '-v', 'error', '-i', self.tmp, '-filter_complex', graph,
                    '-loop', '0', self.out_path])
        if res.returncode != 0:
            raise RuntimeError(f'GIF encoding failed: {res.stderr.decode("utf-8", "ignore").strip()[-400:]}')
        return 'no-audio'

    def _finish_mp4(self) -> str:
        if not self.info.has_audio:
            shutil.move(self.tmp, self.out_path)
            return 'no-audio'
        base = [ffmpeg_exe(), '-y', '-v', 'error', '-i', self.tmp, '-i', self.source_path,
                '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy']
        tail = ['-shortest', '-metadata', 'comment=Contains AI face swap (Swapframe)',
                '-movflags', '+faststart', self.out_path]
        attempts = [['-c:a', 'aac', '-b:a', '192k']]
        if self.info.audio_codec in MP4_SAFE_AUDIO:   # copy losslessly only when every player can play it
            attempts.insert(0, ['-c:a', 'copy'])
        for audio in attempts:
            if _run(base + audio + tail).returncode == 0 and _has_audio(self.out_path):
                return 'audio'
        shutil.move(self.tmp, self.out_path)
        return 'audio-failed'

    def abort(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()
        self.proc.wait()
        _remove(self.tmp)


def _has_audio(path: str) -> bool:
    text = _run([ffmpeg_exe(), '-hide_banner', '-i', path]).stderr.decode('utf-8', 'ignore')
    return bool(re.search(r'Stream #\d+:\d+.*?: Audio:', text))


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
