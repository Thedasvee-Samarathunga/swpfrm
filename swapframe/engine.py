"""Face swap engine: detection, recognition, swapping, masking, enhancement.

Pre/post-processing mirrors FaceFusion's reference implementation so the
HyperSwap / InSwapper / GFPGAN / GPEN ONNX models behave as intended.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from . import models

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
TEMPLATES = {
    'arcface_112_v2': np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                                [0.37097589, 0.82469196], [0.63151696, 0.82325089]], dtype=np.float32),
    'arcface_128': np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                             [0.38710391, 0.72160547], [0.61507734, 0.72034453]], dtype=np.float32),
    'ffhq_512': np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32),
}


def align_matrix(kps: np.ndarray, template: str, size: int) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), TEMPLATES[template] * size,
                                            method=cv2.RANSAC, ransacReprojThreshold=100)
    return matrix


def warp_face(frame: np.ndarray, kps: np.ndarray, template: str, size: int):
    matrix = align_matrix(kps, template, size)
    crop = cv2.warpAffine(frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, matrix


def box_mask(size: int, blur: float = 0.3, padding=(0, 0, 0, 0)) -> np.ndarray:
    blur_size = size * 0.5 * blur
    blur_area = max(int(blur_size / 2), 1)
    mask = np.ones((size, size), np.float32)
    mask[:max(blur_area, int(size * padding[0] / 100)), :] = 0
    mask[-max(blur_area, int(size * padding[2] / 100)):, :] = 0
    mask[:, :max(blur_area, int(size * padding[3] / 100))] = 0
    mask[:, -max(blur_area, int(size * padding[1] / 100)):] = 0
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_size * 0.25)
    return mask


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Blend an aligned crop back into ``frame`` (in place, ROI only) and return it."""
    h, w = frame.shape[:2]
    size = crop.shape[0]
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0], [size, 0], [0, size], [size, size]], np.float32)
    pts = corners @ inverse[:, :2].T + inverse[:, 2]
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return frame
    inverse[0, 2] -= x0
    inverse[1, 2] -= y0
    roi_size = (x1 - x0, y1 - y0)
    inv_mask = cv2.warpAffine(mask, inverse, roi_size).clip(0, 1)[..., None]
    inv_crop = cv2.warpAffine(crop.astype(np.float32), inverse, roi_size, borderMode=cv2.BORDER_REPLICATE)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (inv_mask * inv_crop + (1 - inv_mask) * roi).round().clip(0, 255).astype(np.uint8)
    return frame


def implode_pixel_boost(crop: np.ndarray, total: int, size: int) -> np.ndarray:
    tiles = crop.reshape(size, total, size, total, 3)
    return tiles.transpose(1, 3, 0, 2, 4).reshape(total ** 2, size, size, 3)


def explode_pixel_boost(tiles: Sequence[np.ndarray], total: int, size: int) -> np.ndarray:
    crop = np.stack(tiles, axis=0).reshape(total, total, size, size, 3)
    return crop.transpose(2, 0, 3, 1, 4).reshape(size * total, size * total, 3)


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def available_devices() -> List[str]:
    providers = ort.get_available_providers()
    devices = []
    if 'CUDAExecutionProvider' in providers:
        devices.append('CUDA')
    if 'DmlExecutionProvider' in providers:
        devices.append('DirectML')
    if 'CoreMLExecutionProvider' in providers:
        devices.append('CoreML')
    devices.append('CPU')
    return devices


_PROVIDER = {'CUDA': 'CUDAExecutionProvider', 'DirectML': 'DmlExecutionProvider',
             'CoreML': 'CoreMLExecutionProvider', 'CPU': 'CPUExecutionProvider'}


def create_session(path: str, device: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    providers = ['CPUExecutionProvider']
    if device != 'CPU':
        providers.insert(0, _PROVIDER[device])
    if device == 'DirectML':
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        return ort.InferenceSession(path, options, providers=providers)
    except Exception:
        return ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])


def _np_dtype(onnx_type: str):
    return np.float16 if 'float16' in onnx_type else np.float32


# --------------------------------------------------------------------------- #
# Faces
# --------------------------------------------------------------------------- #
@dataclass
class Face:
    bbox: np.ndarray            # x1, y1, x2, y2
    kps: np.ndarray             # (5, 2)
    score: float
    embedding: Optional[np.ndarray] = None   # raw arcface embedding (512,)
    normed: Optional[np.ndarray] = None      # L2-normalised embedding

    @property
    def area(self) -> float:
        return float(max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1]))


class Detector:
    """RetinaFace / SCRFD style detector (FaceFusion retinaface_10g)."""
    strides = (8, 16, 32)
    anchors = 2

    def __init__(self, session: ort.InferenceSession, size: int = 640):
        self.session = session
        self.size = size
        self.input_name = session.get_inputs()[0].name
        self._centers: dict = {}

    def _anchor_centers(self, stride: int) -> np.ndarray:
        if stride not in self._centers:
            n = self.size // stride
            grid = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            self._centers[stride] = np.stack([centers] * self.anchors, axis=1).reshape(-1, 2)
        return self._centers[stride]

    def detect(self, frame: np.ndarray, threshold: float = 0.5) -> List[Face]:
        h, w = frame.shape[:2]
        scale = min(self.size / h, self.size / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        canvas = np.zeros((self.size, self.size, 3), np.float32)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        blob = ((canvas[:, :, ::-1] - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.session.run(None, {self.input_name: blob})
        if outs[0].ndim == 3:
            outs = [o[0] for o in outs]

        boxes, points, scores = [], [], []
        fmc = len(self.strides)
        for i, stride in enumerate(self.strides):
            score = outs[i].reshape(-1)
            keep = np.where(score >= threshold)[0]
            if not keep.size:
                continue
            centers = self._anchor_centers(stride)[keep]
            dist = outs[i + fmc].reshape(-1, 4)[keep] * stride
            kp = outs[i + fmc * 2].reshape(-1, 10)[keep] * stride
            box = np.stack([centers[:, 0] - dist[:, 0], centers[:, 1] - dist[:, 1],
                            centers[:, 0] + dist[:, 2], centers[:, 1] + dist[:, 3]], axis=-1)
            kps = kp.reshape(-1, 5, 2) + centers[:, None, :]
            boxes.append(box)
            points.append(kps)
            scores.append(score[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        points = np.concatenate(points) / scale
        scores = np.concatenate(scores)

        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), threshold, 0.4)
        keep = np.array(keep).reshape(-1)
        faces = [Face(bbox=boxes[k], kps=points[k], score=float(scores[k])) for k in keep]
        faces.sort(key=lambda f: (f.bbox[0], f.bbox[1]))
        return faces


class Recogniser:
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def embed(self, frame: np.ndarray, face: Face) -> Face:
        crop, _ = warp_face(frame, face.kps, 'arcface_112_v2', 112)
        blob = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        emb = self.session.run(None, {self.input_name: blob})[0].reshape(-1).astype(np.float32)
        face.embedding = emb
        face.normed = emb / (np.linalg.norm(emb) + 1e-8)
        return face


@dataclass(frozen=True)
class SwapperSpec:
    key: str
    kind: str          # 'hyperswap' | 'inswapper'
    template: str
    size: int
    mean: tuple
    std: tuple


SWAPPERS = {
    'hyperswap_1a_256': SwapperSpec('hyperswap_1a_256', 'hyperswap', 'arcface_128', 256, (0.5,) * 3, (0.5,) * 3),
    'hyperswap_1b_256': SwapperSpec('hyperswap_1b_256', 'hyperswap', 'arcface_128', 256, (0.5,) * 3, (0.5,) * 3),
    'hyperswap_1c_256': SwapperSpec('hyperswap_1c_256', 'hyperswap', 'arcface_128', 256, (0.5,) * 3, (0.5,) * 3),
    'inswapper_128_fp16': SwapperSpec('inswapper_128_fp16', 'inswapper', 'arcface_128', 128, (0.0,) * 3, (1.0,) * 3),
}


def load_inswapper_emap(path: str) -> np.ndarray:
    import onnx
    from onnx import numpy_helper
    graph = onnx.load(path).graph
    return numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)


class Swapper:
    def __init__(self, session: ort.InferenceSession, spec: SwapperSpec, emap: Optional[np.ndarray] = None):
        self.session = session
        self.spec = spec
        self.emap = emap
        self.mean = np.array(spec.mean, np.float32)
        self.std = np.array(spec.std, np.float32)
        self.image_input = self.embed_input = None
        for inp in session.get_inputs():
            if len(inp.shape) == 4:
                self.image_input = (inp.name, _np_dtype(inp.type))
            else:
                self.embed_input = (inp.name, _np_dtype(inp.type))
        outputs = session.get_outputs()
        self.output_name = outputs[0].name
        for out in outputs:  # HyperSwap also exports a mask; pick the RGB output
            if len(out.shape) == 4 and out.shape[1] == 3:
                self.output_name = out.name
                break

    def source_embedding(self, faces: Sequence[Face]) -> np.ndarray:
        if self.spec.kind == 'inswapper':
            raw = np.mean([f.embedding for f in faces], axis=0).reshape(1, -1)
            return (raw @ self.emap) / np.linalg.norm(raw)
        normed = np.mean([f.normed for f in faces], axis=0)
        return (normed / (np.linalg.norm(normed) + 1e-8)).reshape(1, -1)

    def _run(self, tile: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        x = tile[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        feed = {self.image_input[0]: x.astype(self.image_input[1]),
                self.embed_input[0]: embedding.astype(self.embed_input[1])}
        out = self.session.run([self.output_name], feed)[0][0].astype(np.float32).transpose(1, 2, 0)
        out = (out * self.std + self.mean).clip(0, 1)
        return out[:, :, ::-1] * 255.0

    def swap_crop(self, crop: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        size = self.spec.size
        total = crop.shape[0] // size
        if total <= 1:
            return self._run(crop, embedding)
        tiles = implode_pixel_boost(crop, total, size)
        return explode_pixel_boost([self._run(t, embedding) for t in tiles], total, size)


class Occluder:
    """XSeg occlusion mask: keeps hands, hair, glasses etc. in front of the new face."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        self.nchw = len(inp.shape) == 4 and inp.shape[1] == 3
        self.size = 256

    def mask(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (self.size, self.size)).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None] if self.nchw else x[None]
        out = np.squeeze(self.session.run(None, {self.input_name: x})[0]).astype(np.float32).clip(0, 1)
        out = cv2.resize(out, crop.shape[:2][::-1])
        return (cv2.GaussianBlur(out.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2


class Enhancer:
    """GFPGAN / GPEN face restoration on an FFHQ-aligned 512 crop."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.mask = box_mask(512, 0.3)

    def enhance(self, frame: np.ndarray, face: Face, blend: float) -> np.ndarray:
        crop, matrix = warp_face(frame, face.kps, 'ffhq_512', 512)
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0][0].astype(np.float32)
        out = ((out.clip(-1, 1) + 1) / 2).transpose(1, 2, 0)[:, :, ::-1] * 255.0
        return paste_back(frame, out, self.mask * float(blend), matrix)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    swapper: str = 'hyperswap_1a_256'
    boost: int = 2                      # pixel boost multiplier (1, 2, 4)
    enhancer: Optional[str] = None      # 'gfpgan_1.4' | 'gpen_bfr_512' | None
    enhance_blend: float = 0.8
    occlusion: bool = False
    mask_blur: float = 0.3
    mask_padding: tuple = (0, 0, 0, 0)
    face_mode: str = 'all'              # 'all' | 'selected' | 'largest'
    reference: List[np.ndarray] = field(default_factory=list)   # normed embeddings
    match_distance: float = 0.6
    detect_threshold: float = 0.5
    stabilise: bool = True


class Pipeline:
    def __init__(self, device: str = 'CPU'):
        self.device = device
        self.lock = threading.RLock()
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._emaps: dict[str, np.ndarray] = {}
        self._mask_cache: dict = {}
        self._track: List[Face] = []

    # -- model management -------------------------------------------------- #
    def set_device(self, device: str) -> None:
        with self.lock:
            if device != self.device:
                self.device = device
                self._sessions.clear()

    def _session(self, key: str, progress=None, cancelled=lambda: False) -> ort.InferenceSession:
        if key not in self._sessions:
            path = models.ensure(key, progress, cancelled)
            device = self.device
            # HyperSwap / XSeg are not CoreML friendly; keep them on CPU on macOS.
            if device == 'CoreML' and key.startswith(('hyperswap', 'xseg', 'inswapper')):
                device = 'CPU'
            self._sessions[key] = create_session(path, device)
        return self._sessions[key]

    def prepare(self, settings: Settings, progress=None, cancelled=lambda: False) -> None:
        """Download + load every model the given settings need."""
        with self.lock:
            for key in self.required_models(settings):
                self._session(key, progress, cancelled)
            if SWAPPERS[settings.swapper].kind == 'inswapper' and settings.swapper not in self._emaps:
                self._emaps[settings.swapper] = load_inswapper_emap(models.model_path(settings.swapper))

    def prepare_core(self, progress=None, cancelled=lambda: False) -> None:
        with self.lock:
            for key in ('retinaface_10g', 'arcface_w600k_r50'):
                self._session(key, progress, cancelled)

    @staticmethod
    def required_models(settings: Settings) -> List[str]:
        keys = ['retinaface_10g', 'arcface_w600k_r50', settings.swapper]
        if settings.enhancer:
            keys.append(settings.enhancer)
        if settings.occlusion:
            keys.append('xseg_1')
        return keys

    @property
    def detector(self) -> Detector:
        return Detector(self._session('retinaface_10g'))

    @property
    def recogniser(self) -> Recogniser:
        return Recogniser(self._session('arcface_w600k_r50'))

    # -- analysis ---------------------------------------------------------- #
    def analyse(self, frame: np.ndarray, threshold: float = 0.5, embed: bool = True) -> List[Face]:
        with self.lock:
            faces = self.detector.detect(frame, threshold)
            if embed:
                rec = self.recogniser
                for f in faces:
                    rec.embed(frame, f)
            return faces

    def source_faces(self, images: Sequence[np.ndarray]) -> List[Face]:
        """Largest face from each source image."""
        result = []
        for img in images:
            faces = self.analyse(img)
            if faces:
                result.append(max(faces, key=lambda f: f.area))
        return result

    def reset_tracking(self) -> None:
        self._track = []

    def _stabilise(self, faces: List[Face]) -> None:
        prev = self._track
        for f in faces:
            best, best_iou = None, 0.0
            for p in prev:
                iou = _iou(f.bbox, p.bbox)
                if iou > best_iou:
                    best, best_iou = p, iou
            if best is not None and best_iou > 0.5:
                size = max(f.bbox[2] - f.bbox[0], 1.0)
                motion = np.abs(f.kps - best.kps).mean() / size
                alpha = float(np.clip(0.35 + motion * 12, 0.35, 1.0))  # move fast -> follow fast
                f.kps = alpha * f.kps + (1 - alpha) * best.kps
        self._track = [Face(f.bbox.copy(), f.kps.copy(), f.score) for f in faces]

    # -- processing -------------------------------------------------------- #
    def _crop_mask(self, size: int, settings: Settings) -> np.ndarray:
        key = (size, settings.mask_blur, tuple(settings.mask_padding))
        if key not in self._mask_cache:
            self._mask_cache[key] = box_mask(size, settings.mask_blur, settings.mask_padding)
        return self._mask_cache[key]

    def process(self, frame: np.ndarray, source: Sequence[Face], settings: Settings,
                video: bool = False) -> np.ndarray:
        """Swap faces in ``frame`` (BGR uint8). Returns a new frame."""
        if not source:
            return frame
        with self.lock:
            spec = SWAPPERS[settings.swapper]
            swapper = Swapper(self._session(settings.swapper), spec, self._emaps.get(settings.swapper))
            embedding = swapper.source_embedding(source)
            occluder = Occluder(self._session('xseg_1')) if settings.occlusion else None
            enhancer = Enhancer(self._session(settings.enhancer)) if settings.enhancer else None

            need_identity = settings.face_mode == 'selected' and settings.reference
            faces = self.analyse(frame, settings.detect_threshold, embed=bool(need_identity))
            if video and settings.stabilise:
                self._stabilise(faces)
            faces = self._select(faces, settings)

            out = frame.copy()
            crop_size = spec.size * max(1, int(settings.boost))
            base_mask = self._crop_mask(crop_size, settings)
            for face in faces:
                crop, matrix = warp_face(out, face.kps, spec.template, crop_size)
                mask = base_mask
                if occluder is not None:
                    mask = np.minimum(mask, occluder.mask(crop))
                swapped = swapper.swap_crop(crop, embedding)
                paste_back(out, swapped, mask, matrix)
                if enhancer is not None:
                    enhancer.enhance(out, face, settings.enhance_blend)
            return out

    @staticmethod
    def _select(faces: List[Face], settings: Settings) -> List[Face]:
        if not faces:
            return faces
        if settings.face_mode == 'largest':
            return [max(faces, key=lambda f: f.area)]
        if settings.face_mode == 'selected':
            if not settings.reference:
                return []
            refs = np.stack(settings.reference)
            picked = []
            for f in faces:
                if f.normed is None:
                    continue
                distance = 1.0 - float(np.max(refs @ f.normed))
                if distance < settings.match_distance:
                    picked.append(f)
            return picked
        return faces


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


ProgressFn = Callable[[str, int, int], None]
