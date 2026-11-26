"""Face swap engine: detection, landmarks, recognition, swapping, expression, masking, colour, enhancement.

Pre/post-processing mirrors FaceFusion's reference implementation for the
HyperSwap / InSwapper / LivePortrait / GFPGAN / GPEN / XSeg / BiSeNet / 2DFAN4
ONNX models. AlphaFace's input conventions are detected automatically the first
time it is used (see ``calibrate``) and cached.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Sequence

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


def match_colour(swapped: np.ndarray, target: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Shift the swapped face's skin tone / lighting toward the original face (LAB statistics inside ``mask``)."""
    weight = mask > 0.5
    if weight.sum() < 64 or strength <= 0:
        return swapped
    src = cv2.cvtColor(swapped.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    s, r = src[weight], ref[weight]
    s_mean, s_std = s.mean(0), s.std(0) + 1e-3
    r_mean, r_std = r.mean(0), r.std(0) + 1e-3
    ratio = np.clip(r_std / s_std, 0.75, 1.33)
    ratio[1:] = 1.0  # keep the source face's own chroma contrast; only match tone + lighting contrast
    out = (src - s_mean) * ratio + r_mean
    out = src + (out - src) * float(strength)
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Extrinsic x-y-z Euler rotation in degrees (same as scipy ``from_euler('xyz', ..., degrees=True)``)."""
    p, y, r = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def _angle(value: np.ndarray) -> float:
    """Head angle in degrees; also accepts LivePortrait's raw 66-bin logits."""
    v = np.asarray(value, np.float32).reshape(-1)
    if v.size >= 66:
        e = np.exp(v - v.max())
        prob = e / e.sum()
        return float((prob * np.arange(v.size)).sum() * 3 - 97.5)
    return float(v[0])


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
    devices.append('CPU')
    return devices


_PROVIDER = {'CUDA': 'CUDAExecutionProvider', 'DirectML': 'DmlExecutionProvider', 'CPU': 'CPUExecutionProvider'}


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
    landmarks68: Optional[np.ndarray] = None

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

    def detect_rotated(self, frame: np.ndarray, threshold: float) -> List[Face]:
        """Retry on 90/270/180 degree rotations (lying down, upside-down, sideways phone shots)."""
        h, w = frame.shape[:2]
        for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            faces = self.detect(cv2.rotate(frame, code), threshold)
            if not faces:
                continue
            for f in faces:
                f.kps = _unrotate(f.kps, code, w, h)
                pts = _unrotate(f.bbox.reshape(2, 2), code, w, h)
                f.bbox = np.concatenate([pts.min(0), pts.max(0)])
            return faces
        return []

    def detect_region(self, frame: np.ndarray, bbox: np.ndarray, threshold: float) -> List[Face]:
        """Zoomed re-detection around where a face was last seen (recovers turned / small faces)."""
        h, w = frame.shape[:2]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 2.2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        faces = self.detect(frame[y0:y1, x0:x1], threshold)
        for f in faces:
            f.bbox = f.bbox + [x0, y0, x0, y0]
            f.kps = f.kps + [x0, y0]
        return faces


def _unrotate(pts: np.ndarray, code: int, w: int, h: int) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    if code == cv2.ROTATE_90_CLOCKWISE:
        return np.stack([y, h - 1 - x], axis=1)
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return np.stack([w - 1 - y, x], axis=1)
    return np.stack([w - 1 - x, h - 1 - y], axis=1)


class Landmarker:
    """2DFAN4 68-point landmarks -> refined 5-point alignment (steadier on turned heads + expressions)."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def refine(self, frame: np.ndarray, face: Face) -> Face:
        bbox = face.bbox
        scale = 195.0 / max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
        tx, ty = (256 - (bbox[2:] + bbox[:2]) * scale) * 0.5
        matrix = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
        crop = cv2.warpAffine(frame, matrix, (256, 256))
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
        if lab[:, :, 0].mean() < 30:  # dark shots: equalise so landmarks still lock on
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(lab[:, :, 0])
            crop = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        x = crop.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = self.session.run(None, {self.input_name: x})
        coords = next(o for o in outs if o.ndim == 3)
        heat = next((o for o in outs if o.ndim == 4), None)
        pts = coords[0, :, :2] / 64.0 * 256.0
        inv = cv2.invertAffineTransform(matrix)
        pts = pts @ inv[:, :2].T + inv[:, 2]
        confidence = float(np.interp(heat.max(axis=(2, 3)).mean(), [0, 0.9], [0, 1])) if heat is not None else 1.0
        if confidence < 0.5:
            return face
        kps5 = np.stack([pts[36:42].mean(0), pts[42:48].mean(0), pts[30], pts[48], pts[54]]).astype(np.float32)
        size = max(bbox[2] - bbox[0], 1.0)
        if np.abs(kps5 - face.kps).mean() / size > 0.12:  # disagree badly -> trust the detector
            return face
        face.kps = kps5
        face.landmarks68 = pts.astype(np.float32)
        return face


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
class SwapConfig:
    template: str
    mean: tuple
    std: tuple
    embed: str          # 'normed' | 'emap'


KNOWN = {
    'hyperswap_1a_256': SwapConfig('arcface_128', (0.5,) * 3, (0.5,) * 3, 'normed'),
    'hyperswap_1b_256': SwapConfig('arcface_128', (0.5,) * 3, (0.5,) * 3, 'normed'),
    'hyperswap_1c_256': SwapConfig('arcface_128', (0.5,) * 3, (0.5,) * 3, 'normed'),
    'inswapper_128_fp16': SwapConfig('arcface_128', (0.0,) * 3, (1.0,) * 3, 'emap'),
}
AUTO = {'alphaface_256'}  # conventions detected on first use
SWAPPER_KEYS = list(KNOWN) + sorted(AUTO)
EXPRESSION_MODELS = ('live_portrait_feature_extractor', 'live_portrait_motion_extractor', 'live_portrait_generator')


def find_emap(path: str) -> Optional[np.ndarray]:
    """Embedding projection matrix stored as the last 512x512 initializer (InSwapper / AlphaFace style)."""
    import onnx
    from onnx import numpy_helper
    graph = onnx.load(path, load_external_data=False).graph
    for init in reversed(graph.initializer):
        if list(init.dims) == [512, 512]:
            return numpy_helper.to_array(init).astype(np.float32)
    return None


class Swapper:
    def __init__(self, session: ort.InferenceSession, config: SwapConfig, emap: Optional[np.ndarray] = None):
        self.session = session
        self.config = config
        self.emap = emap
        self.mean = np.array(config.mean, np.float32)
        self.std = np.array(config.std, np.float32)
        self.image_input = self.embed_input = None
        self.size = 256
        for inp in session.get_inputs():
            if len(inp.shape) == 4:
                self.image_input = (inp.name, _np_dtype(inp.type))
                if isinstance(inp.shape[2], int):
                    self.size = inp.shape[2]
            else:
                self.embed_input = (inp.name, _np_dtype(inp.type))
        outputs = session.get_outputs()
        self.output_name = outputs[0].name
        for out in outputs:  # some swappers also export a mask; pick the RGB output
            if len(out.shape) == 4 and out.shape[1] == 3:
                self.output_name = out.name
                break

    def source_embedding(self, faces: Sequence[Face]) -> np.ndarray:
        if self.config.embed == 'emap' and self.emap is not None:
            raw = np.mean([f.embedding for f in faces], axis=0).reshape(1, -1)
            return (raw @ self.emap) / (np.linalg.norm(raw) + 1e-8)
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
        total = crop.shape[0] // self.size
        if total <= 1:
            return self._run(crop, embedding)
        tiles = implode_pixel_boost(crop, total, self.size)
        return explode_pixel_boost([self._run(t, embedding) for t in tiles], total, self.size)


class ExpressionRestorer:
    """LivePortrait: re-animates the swapped face with the original target expression.

    Mirrors FaceFusion's expression_restorer: the motion extractor reads the expression of the
    original face and of the swapped face, the generator re-renders the swapped face's appearance
    with the original's expression (eyes, brows, mouth), while head pose comes from the swap.
    """
    SIZE = 512                       # crop size (arcface_128 template)
    KEEP_FROM_SWAP = [0, 4, 5, 8, 9]  # keypoints that stay with the swapped face (FaceFusion)

    def __init__(self, feature, motion, generator):
        self.feature, self.motion, self.generator = feature, motion, generator
        self.f_in = feature.get_inputs()[0].name
        self.m_in = motion.get_inputs()[0].name
        names = [i.name for i in generator.get_inputs()]
        if {'feature_volume', 'source', 'target'} <= set(names):
            self.g_names = ('feature_volume', 'source', 'target')
        else:
            self.g_names = tuple(names[:3])

    @staticmethod
    def _prepare(crop: np.ndarray) -> np.ndarray:
        half = ExpressionRestorer.SIZE // 2
        x = cv2.resize(crop, (half, half), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def _motion(self, x: np.ndarray):
        pitch, yaw, roll, scale, translation, expression, points = self.motion.run(None, {self.m_in: x})[:7]
        expression = np.asarray(expression, np.float32).reshape(1, -1, 3)
        points = np.asarray(points, np.float32).reshape(1, -1, 3)
        scale = np.asarray(scale, np.float32).reshape(-1)[0]
        translation = np.asarray(translation, np.float32).reshape(-1)[:3].copy()
        if translation.size == 3:
            translation[2] = 0.0  # LivePortrait only translates in x/y
        else:
            translation = np.pad(translation, (0, 3 - translation.size))
        return _angle(pitch), _angle(yaw), _angle(roll), scale, translation, expression, points

    def restore(self, original_crop: np.ndarray, swapped_crop: np.ndarray, factor: float) -> np.ndarray:
        """Both crops are BGR SIZE x SIZE aligned with the arcface_128 template. Returns BGR float32."""
        src = self._prepare(original_crop)
        tgt = self._prepare(swapped_crop)
        volume = self.feature.run(None, {self.f_in: tgt})[0]
        original_expression = self._motion(src)[5]
        pitch, yaw, roll, scale, translation, swapped_expression, points = self._motion(tgt)
        rot = rotation_matrix(pitch, yaw, roll)
        expression = original_expression.copy()
        expression[:, self.KEEP_FROM_SWAP] = swapped_expression[:, self.KEEP_FROM_SWAP]
        expression = expression * factor + swapped_expression * (1 - factor)
        # keep it plausible: never push a keypoint wildly further than either face did
        expression = np.clip(expression, np.minimum(original_expression, swapped_expression) - 0.05,
                             np.maximum(original_expression, swapped_expression) + 0.05)
        base = points @ rot.T
        source_points = (scale * (base + expression) + translation).astype(np.float32)
        target_points = (scale * (base + swapped_expression) + translation).astype(np.float32)
        out = self.generator.run(None, {self.g_names[0]: volume, self.g_names[1]: source_points,
                                        self.g_names[2]: target_points})[0][0]
        out = out.astype(np.float32).transpose(1, 2, 0).clip(0, 1)[:, :, ::-1] * 255.0
        if out.shape[0] != self.SIZE:
            out = cv2.resize(out, (self.SIZE, self.SIZE), interpolation=cv2.INTER_CUBIC)
        return out


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


class FaceParser:
    """BiSeNet face parsing -> soft mask that follows the real face outline (no box edges on turned heads)."""
    REGIONS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]  # skin, brows, eyes, glasses, nose, mouth, lips
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def mask(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0][0]
        region = np.isin(out.argmax(0), self.REGIONS).astype(np.float32)
        region = cv2.dilate(region, np.ones((9, 9), np.uint8))
        region = cv2.resize(region, crop.shape[:2][::-1])
        return (cv2.GaussianBlur(region.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2


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
# Auto-calibration for models whose I/O conventions aren't hard-coded
# --------------------------------------------------------------------------- #
def _calibration_file() -> str:
    return os.path.join(models.models_dir(), 'calibration.json')


def _load_calibration() -> dict:
    try:
        with open(_calibration_file(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_calibration(data: dict) -> None:
    try:
        with open(_calibration_file(), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def calibrate(session, emap, frame: np.ndarray, target: Face, source: Sequence[Face],
              recogniser: Recogniser) -> SwapConfig:
    """Try every plausible convention and keep the one that best (a) carries the source identity and
    (b) keeps the target's lighting/colour. Wrong conventions fail one of the two very obviously."""
    best, best_score = None, -1e9
    embeds = ['normed'] + (['emap'] if emap is not None else [])
    for template, norm, embed in product(('arcface_128', 'ffhq_512'), ((0.5, 0.5), (0.0, 1.0)), embeds):
        cfg = SwapConfig(template, (norm[0],) * 3, (norm[1],) * 3, embed)
        sw = Swapper(session, cfg, emap)
        try:
            crop, matrix = warp_face(frame, target.kps, template, sw.size)
            out = sw.swap_crop(crop, sw.source_embedding(source))
        except Exception:
            continue
        if not np.isfinite(out).all():
            continue
        inner = box_mask(sw.size, 0.3) > 0.5
        colour_err = float(np.abs(cv2.blur(out, (15, 15)) - cv2.blur(crop.astype(np.float32), (15, 15)))[inner].mean()) / 255
        test = paste_back(frame.copy(), out, box_mask(sw.size, 0.3), matrix)
        probe = Face(target.bbox.copy(), target.kps.copy(), target.score)
        recogniser.embed(test, probe)
        identity = float(np.mean([probe.normed @ s.normed for s in source]))
        score = identity - 1.5 * colour_err
        if score > best_score:
            best, best_score = cfg, score
    return best or SwapConfig('arcface_128', (0.5,) * 3, (0.5,) * 3, 'normed')


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    swapper: str = 'alphaface_256'
    boost: int = 2                      # pixel boost multiplier (1, 2, 4)
    expression: float = 0.0             # 0 = off, 1 = fully the target's original expression
    enhancer: Optional[str] = None      # 'gfpgan_1.4' | 'gpen_bfr_512' | None
    enhance_blend: float = 0.8
    occlusion: bool = False
    face_shape: bool = True             # BiSeNet face-outline mask
    landmarks: bool = True              # 68-point alignment
    colour_match: float = 0.8           # 0 = off
    mask_blur: float = 0.3
    mask_padding: tuple = (0, 0, 0, 0)
    face_mode: str = 'all'              # 'all' | 'selected' | 'largest'
    reference: List[np.ndarray] = field(default_factory=list)   # normed embeddings
    match_distance: float = 0.6
    detect_threshold: float = 0.5
    stabilise: bool = True


@dataclass
class _Track:
    face: Face
    misses: int = 0


class Pipeline:
    def __init__(self, device: str = 'CPU'):
        self.device = device
        self.lock = threading.RLock()
        self._sessions: dict = {}
        self._emaps: dict = {}
        self._configs: dict = {}
        self._mask_cache: dict = {}
        self._tracks: List[_Track] = []

    # -- model management -------------------------------------------------- #
    def set_device(self, device: str) -> None:
        with self.lock:
            if device != self.device:
                self.device = device
                self._sessions.clear()

    def _session(self, key: str, progress=None, cancelled=lambda: False):
        if key not in self._sessions:
            path = models.ensure(key, progress, cancelled)
            self._sessions[key] = create_session(path, self.device)
        return self._sessions[key]

    def prepare_core(self, progress=None, cancelled=lambda: False) -> None:
        with self.lock:
            for key in ('retinaface_10g', 'arcface_w600k_r50', '2dfan4'):
                self._session(key, progress, cancelled)

    def prepare(self, settings: Settings, progress=None, cancelled=lambda: False) -> None:
        """Download + load every model the given settings need."""
        with self.lock:
            for key in self.required_models(settings):
                self._session(key, progress, cancelled)
            key = settings.swapper
            if key not in self._emaps:
                needs = key in AUTO or KNOWN.get(key, KNOWN['hyperswap_1a_256']).embed == 'emap'
                self._emaps[key] = find_emap(models.model_path(key)) if needs and models.is_downloaded(key) else None

    @staticmethod
    def required_models(settings: Settings) -> List[str]:
        keys = ['retinaface_10g', 'arcface_w600k_r50', settings.swapper]
        if settings.landmarks:
            keys.append('2dfan4')
        if settings.expression > 0:
            keys.extend(EXPRESSION_MODELS)
        if settings.enhancer:
            keys.append(settings.enhancer)
        if settings.occlusion:
            keys.append('xseg_1')
        if settings.face_shape:
            keys.append('bisenet_resnet_34')
        return keys

    def _config(self, key: str, frame: np.ndarray, target: Face, source: Sequence[Face]) -> SwapConfig:
        if key in KNOWN:
            return KNOWN[key]
        if key in self._configs:
            return self._configs[key]
        stamp = str(os.path.getsize(models.model_path(key))) if models.is_downloaded(key) else 'mem'
        cache = _load_calibration()
        hit = cache.get(key)
        if hit and hit.get('stamp') == stamp:
            cfg = SwapConfig(hit['template'], tuple(hit['mean']), tuple(hit['std']), hit['embed'])
        else:
            cfg = calibrate(self._sessions[key], self._emaps.get(key), frame, target, source, self.recogniser)
            cache[key] = dict(stamp=stamp, template=cfg.template, mean=list(cfg.mean), std=list(cfg.std),
                              embed=cfg.embed)
            _save_calibration(cache)
        self._configs[key] = cfg
        return cfg

    @property
    def detector(self) -> Detector:
        return Detector(self._session('retinaface_10g'))

    @property
    def recogniser(self) -> Recogniser:
        return Recogniser(self._session('arcface_w600k_r50'))

    # -- analysis ---------------------------------------------------------- #
    def analyse(self, frame: np.ndarray, threshold: float = 0.5, embed: bool = True,
                landmarks: bool = True, rotate: bool = True) -> List[Face]:
        with self.lock:
            det = self.detector
            faces = det.detect(frame, threshold)
            if not faces and rotate:
                faces = det.detect_rotated(frame, threshold)
            self._finish(frame, faces, embed, landmarks)
            return faces

    def _finish(self, frame, faces, embed, landmarks):
        if landmarks:
            try:
                lm = Landmarker(self._session('2dfan4'))
                for f in faces:
                    lm.refine(frame, f)
            except Exception:
                pass
        if embed:
            rec = self.recogniser
            for f in faces:
                rec.embed(frame, f)

    def source_faces(self, images: Sequence[np.ndarray]) -> List[Face]:
        """Largest face from each source image."""
        result = []
        for img in images:
            faces = self.analyse(img)
            if faces:
                result.append(max(faces, key=lambda f: f.area))
        return result

    def reset_tracking(self) -> None:
        self._tracks = []

    def _track(self, frame: np.ndarray, faces: List[Face], settings: Settings) -> List[Face]:
        """Smooth landmarks between frames and recover faces the detector briefly loses (head turns, blur)."""
        matched = set()
        for f in faces:
            best, best_iou = None, 0.0
            for i, t in enumerate(self._tracks):
                iou = _iou(f.bbox, t.face.bbox)
                if iou > best_iou and i not in matched:
                    best, best_iou = i, iou
            if best is not None and best_iou > 0.3:
                matched.add(best)
                if settings.stabilise:
                    prev = self._tracks[best].face
                    size = max(f.bbox[2] - f.bbox[0], 1.0)
                    motion = np.abs(f.kps - prev.kps).mean() / size
                    alpha = float(np.clip(0.35 + motion * 12, 0.35, 1.0))  # fast motion -> follow fast
                    f.kps = alpha * f.kps + (1 - alpha) * prev.kps
        recovered = []
        det = self.detector
        for i, t in enumerate(self._tracks):
            if i in matched or t.misses >= 6:
                continue
            found = det.detect_region(frame, t.face.bbox, max(0.25, settings.detect_threshold - 0.25))
            found = [g for g in found if _iou(g.bbox, t.face.bbox) > 0.15]
            if found:
                g = max(found, key=lambda g: _iou(g.bbox, t.face.bbox))
                self._finish(frame, [g], embed=False, landmarks=settings.landmarks)
                g.normed = t.face.normed
                recovered.append(g)
        new_tracks = [_Track(_copy(f)) for f in faces + recovered]
        for i, t in enumerate(self._tracks):
            if i not in matched and not any(_iou(r.bbox, t.face.bbox) > 0.15 for r in recovered):
                if t.misses < 6:
                    new_tracks.append(_Track(t.face, t.misses + 1))
        self._tracks = new_tracks
        return faces + recovered

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
            need_identity = settings.face_mode == 'selected' and settings.reference
            faces = self.analyse(frame, settings.detect_threshold, embed=bool(need_identity),
                                 landmarks=settings.landmarks, rotate=not video)
            if video:
                faces = self._track(frame, faces, settings)
            faces = self._select(faces, settings)
            if not faces:
                return frame

            key = settings.swapper
            cfg = self._config(key, frame, max(faces, key=lambda f: f.area), source)
            swapper = Swapper(self._session(key), cfg, self._emaps.get(key))
            embedding = swapper.source_embedding(source)
            occluder = Occluder(self._session('xseg_1')) if settings.occlusion else None
            parser = FaceParser(self._session('bisenet_resnet_34')) if settings.face_shape else None
            restorer = ExpressionRestorer(*(self._session(k) for k in EXPRESSION_MODELS)) \
                if settings.expression > 0 else None
            enhancer = Enhancer(self._session(settings.enhancer)) if settings.enhancer else None

            out = frame.copy()
            crop_size = swapper.size * max(1, int(settings.boost))
            base_mask = self._crop_mask(crop_size, settings)
            for face in faces:
                crop, matrix = warp_face(out, face.kps, cfg.template, crop_size)
                mask = base_mask
                if parser is not None:
                    mask = np.minimum(mask, parser.mask(crop))
                if occluder is not None:
                    mask = np.minimum(mask, occluder.mask(crop))
                swapped = swapper.swap_crop(crop, embedding)
                if settings.colour_match > 0:
                    swapped = match_colour(swapped, crop, mask, settings.colour_match)
                paste_back(out, swapped, mask, matrix)
                if restorer is not None:
                    self._restore_expression(restorer, frame, out, face, settings, parser, occluder)
                if enhancer is not None:
                    enhancer.enhance(out, face, settings.enhance_blend)
            return out

    def _restore_expression(self, restorer: ExpressionRestorer, original: np.ndarray, out: np.ndarray,
                            face: Face, settings: Settings, parser, occluder) -> None:
        size = ExpressionRestorer.SIZE
        original_crop, _ = warp_face(original, face.kps, 'arcface_128', size)
        swapped_crop, matrix = warp_face(out, face.kps, 'arcface_128', size)
        restored = restorer.restore(original_crop, swapped_crop, min(1.2, settings.expression * 1.2))
        mask = self._crop_mask(size, settings)
        if parser is not None:
            mask = np.minimum(mask, parser.mask(original_crop))
        if occluder is not None:
            mask = np.minimum(mask, occluder.mask(original_crop))
        if settings.colour_match > 0:
            restored = match_colour(restored, swapped_crop, mask, 1.0)
        paste_back(out, restored, mask, matrix)

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
            return [f for f in faces
                    if f.normed is not None and 1.0 - float(np.max(refs @ f.normed)) < settings.match_distance]
        return faces


def _copy(f: Face) -> Face:
    return Face(f.bbox.copy(), f.kps.copy(), f.score, normed=f.normed)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
