"""Offline tests: exercise the whole pipeline with fake ONNX sessions (no downloads)."""
import os, sys, tempfile, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['SWAPFRAME_MODELS'] = tempfile.mkdtemp()
from swapframe import engine, media
from swapframe.engine import (Pipeline, Settings, implode_pixel_boost, explode_pixel_boost, paste_back,
                              match_colour, calibrate, Landmarker, ExpressionRestorer, rotation_matrix, _angle,
                              skin_stats, tone_body, Face)


class IO:
    def __init__(s, name, shape, type='tensor(float)'): s.name, s.shape, s.type = name, shape, type


class FakeDetector:
    """Emits one face in SCRFD output layout around the face-coloured ellipse."""
    def get_inputs(s): return [IO('input.1', [1, 3, '?', '?'])]
    def run(s, _, feed):
        x = feed['input.1'][0]
        size = x.shape[1]
        ys, xs = np.nonzero((x[0] - x[2]) > 0.25)   # red-ish minus blue-ish = face blob (RGB order)
        outs_s, outs_b, outs_k = [], [], []
        for stride in (8, 16, 32):
            n = (size // stride) ** 2 * 2
            sc = np.zeros((n, 1), np.float32); bb = np.zeros((n, 4), np.float32); kp = np.zeros((n, 10), np.float32)
            if stride == 16 and len(xs) > 50:
                box = np.array([np.percentile(xs, 1), np.percentile(ys, 1), np.percentile(xs, 99),
                                min(np.percentile(ys, 99), np.percentile(ys, 1) + (np.percentile(xs, 99) - np.percentile(xs, 1)) * 1.3)],
                               np.float32)
                w, h = box[2] - box[0], box[3] - box[1]
                kps = np.array([[.35, .4], [.65, .4], [.5, .55], [.38, .72], [.62, .72]], np.float32) * [w, h] + box[:2]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                gx, gy = int(cx // 16), int(cy // 16)
                idx = (gy * (size // 16) + gx) * 2
                ax, ay = gx * 16, gy * 16
                sc[idx] = 0.95
                bb[idx] = [(ax - box[0]) / 16, (ay - box[1]) / 16, (box[2] - ax) / 16, (box[3] - ay) / 16]
                kp[idx] = ((kps - [ax, ay]) / 16).reshape(-1)
            outs_s.append(sc); outs_b.append(bb); outs_k.append(kp)
        return outs_s + outs_b + outs_k


class FakeRec:
    def get_inputs(s): return [IO('input', [1, 3, 112, 112])]
    def run(s, _, feed):
        c = feed['input'][0, :, 40:80, 40:80].mean(axis=(1, 2))
        v = np.zeros(512, np.float32); v[:3] = c; v[3] = 0.05
        return [v[None]]


class FakeSwap:
    def __init__(s, size=256): s.calls = 0; s.size = size
    def get_inputs(s): return [IO('source', [1, 512]), IO('target', [1, 3, s.size, s.size])]
    def get_outputs(s): return [IO('mask', [1, 1, s.size, s.size]), IO('output', [1, 3, s.size, s.size])]
    def run(s, names, feed):
        s.calls += 1
        t = feed['target']; e = feed['source'][0]
        img = t * 0.5 + 0.5
        col = (e[:3] / (np.abs(e[:3]).max() + 1e-8) * 0.5 + 0.5).reshape(1, 3, 1, 1)   # paint the identity colour
        out = np.clip(img * 0.3 + col * 0.7, 0, 1)
        return [(out - 0.5) / 0.5]


class Ident:
    def __init__(s, kind): s.kind = kind
    def get_inputs(s):
        return [IO('input', {'enh': [1, 3, 512, 512], 'xseg': [1, 256, 256, 3], 'parse': [1, 3, 512, 512],
                             'lm': [1, 3, 256, 256]}[s.kind])]
    def run(s, _, feed):
        x = feed['input']
        if s.kind == 'enh': return [x]
        if s.kind == 'xseg': return [np.ones((1, 256, 256, 1), np.float32)]
        if s.kind == 'parse':
            o = np.zeros((1, 19, 512, 512), np.float32); o[0, 0] = 1; o[0, 1, 96:416, 96:416] = 2
            return [o]
        pts = np.zeros((1, 68, 3), np.float32); pts[0, :, :2] = 32
        pts[0, 36:42, :2] = [24, 26]; pts[0, 42:48, :2] = [40, 26]; pts[0, 30, :2] = [32, 34]
        pts[0, 48, :2] = [26, 44]; pts[0, 54, :2] = [38, 44]
        heat = np.zeros((1, 68, 64, 64), np.float32); heat[..., 0, 0] = 0.9
        return [pts, heat]


class FakeFeature:
    def get_inputs(s): return [IO('input', [1, 3, 256, 256])]
    def run(s, _, feed):
        assert feed['input'].shape == (1, 3, 256, 256) and feed['input'].max() <= 1.0
        return [feed['input'][:, :, ::32, ::32][:, :, None].astype(np.float32)]


class FakeMotion:
    def get_inputs(s): return [IO('input', [1, 3, 256, 256])]
    def run(s, _, feed):
        x = feed['input'][0]
        mouth = float(x[:, 170:220, 90:170].mean())
        exp = np.full((1, 21, 3), mouth * 0.1, np.float32)
        pts = np.linspace(-0.5, 0.5, 63, dtype=np.float32).reshape(1, 21, 3)
        logits = np.zeros((1, 66), np.float32); logits[0, 32] = 50
        return [logits, np.array([[3.0]], np.float32), np.array([[0.0]], np.float32), np.array([[1.2]], np.float32),
                np.array([[0.01, 0.02, 0.5]], np.float32), exp, pts]


class FakeGenerator:
    def get_inputs(s): return [IO('feature_volume', [1, 3, 1, 8, 8]), IO('source', [1, 21, 3]), IO('target', [1, 21, 3])]
    def run(s, _, feed):
        delta = float((feed['source'] - feed['target']).mean())
        vol = feed['feature_volume'][0, :, 0]
        img = cv2.resize(vol.transpose(1, 2, 0), (512, 512)).transpose(2, 0, 1)
        img[:, 340:440, 180:340] = np.clip(img[:, 340:440, 180:340] + delta * 40, 0, 1)
        return [img[None].astype(np.float32)]


SKIN_TGT = (90, 140, 200)     # light skin (BGR)
SKIN_SRC = (60, 95, 150)      # deeper skin (BGR)


def scene(mouth=60, skin=SKIN_TGT, body=True):
    frame = np.full((480, 640, 3), (120, 60, 20), np.uint8)   # blue-ish background (not skin)
    if body:
        cv2.rectangle(frame, (290, 330), (350, 480), skin, -1)              # neck
        cv2.rectangle(frame, (150, 400), (220, 480), skin, -1)              # hand
    cv2.ellipse(frame, (320, 230), (80, 100), 0, 0, 360, skin, -1)
    cv2.ellipse(frame, (320, 280), (30, 12), 0, 0, 360, (mouth,) * 3, -1)
    return frame


def build(swap=None):
    p = Pipeline('CPU')
    p._sessions = {'retinaface_10g': FakeDetector(), 'arcface_w600k_r50': FakeRec(),
                   'alphaface_256': swap or FakeSwap(), 'hyperswap_1a_256': swap or FakeSwap(),
                   'gfpgan_1.4': Ident('enh'), 'xseg_1': Ident('xseg'), 'bisenet_resnet_34': Ident('parse'),
                   '2dfan4': Ident('lm'), 'live_portrait_feature_extractor': FakeFeature(),
                   'live_portrait_motion_extractor': FakeMotion(), 'live_portrait_generator': FakeGenerator()}
    p._emaps = {'alphaface_256': None, 'hyperswap_1a_256': None}
    return p


def source_face(p, skin=(20, 20, 230)):
    img = np.full((300, 300, 3), (120, 60, 20), np.uint8)
    cv2.ellipse(img, (150, 150), (60, 75), 0, 0, 360, skin, -1)
    return p.source_faces([img])


def test_pixel_boost_roundtrip():
    crop = np.random.rand(512, 512, 3).astype(np.float32)
    assert np.allclose(explode_pixel_boost(list(implode_pixel_boost(crop, 2, 256)), 2, 256), crop)


def test_paste_back_zero_mask_is_noop():
    frame = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    kps = np.array([[170, 130], [230, 130], [200, 165], [178, 200], [222, 200]], np.float32)
    crop, m = engine.warp_face(frame, kps, 'arcface_128', 256)
    assert np.array_equal(paste_back(frame.copy(), crop.astype(np.float32), np.zeros((256, 256), np.float32), m), frame)


def test_colour_match_moves_toward_target():
    tgt = np.full((64, 64, 3), (60, 120, 180), np.float32)
    sw = np.full((64, 64, 3), (120, 120, 120), np.float32)
    out = match_colour(sw, tgt, np.ones((64, 64), np.float32), 1.0)
    assert np.abs(out - tgt).mean() < np.abs(sw - tgt).mean() / 3


def test_rotation_and_angles():
    assert np.allclose(rotation_matrix(0, 0, 0), np.eye(3))
    assert np.allclose(rotation_matrix(0, 90, 0) @ [1, 0, 0], [0, 0, -1], atol=1e-6)
    assert abs(_angle(np.array([[12.5]])) - 12.5) < 1e-6
    logits = np.zeros(66); logits[40] = 80
    assert abs(_angle(logits) - (40 * 3 - 97.5)) < 1e-3


def test_expression_restorer_moves_toward_original():
    r = ExpressionRestorer(FakeFeature(), FakeMotion(), FakeGenerator())
    out = r.restore(np.full((512, 512, 3), 200, np.uint8), np.full((512, 512, 3), 60, np.uint8), 1.0)
    assert out.shape == (512, 512, 3) and np.isfinite(out).all()
    assert out[340:440, 180:340].mean() > out[:100].mean() + 1


def test_skin_stats_and_body_tone():
    crop = np.full((256, 256, 3), SKIN_TGT, np.uint8)
    stats = skin_stats(crop)
    assert stats is not None
    lab = cv2.cvtColor(np.uint8([[SKIN_TGT]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    assert np.abs(stats[0] - lab).max() < 2
    src = skin_stats(np.full((256, 256, 3), SKIN_SRC, np.uint8))
    frame = scene()
    before = frame.copy()
    face = Face(np.array([240, 130, 400, 330], np.float32), np.zeros((5, 2), np.float32), 1.0)
    tone_body(frame, face, stats, src, 1.0)
    neck_b, neck_a = before[420, 320].astype(int), frame[420, 320].astype(int)
    assert np.abs(neck_a - np.array(SKIN_SRC)).sum() < np.abs(neck_b - np.array(SKIN_SRC)).sum() / 2, (neck_b, neck_a)
    bg_b, bg_a = before[50, 600].astype(int), frame[50, 600].astype(int)
    assert np.abs(bg_a - bg_b).max() <= 2, 'background must not be recoloured'
    hand = frame[440, 185].astype(int)
    assert np.abs(hand - np.array(SKIN_SRC)).sum() < np.abs(before[440, 185].astype(int) - np.array(SKIN_SRC)).sum()


def test_body_tone_in_pipeline():
    p = build()
    src = source_face(p, skin=SKIN_SRC)
    assert src and src[0].crop is not None
    frame = scene()
    off = p.process(frame, src, Settings(body_tone=0.0))
    on = p.process(frame, src, Settings(body_tone=1.0))
    assert np.array_equal(off[440, 320], frame[440, 320])
    assert np.abs(on[440, 320].astype(int) - SKIN_SRC).sum() < np.abs(off[440, 320].astype(int) - SKIN_SRC).sum()
    assert np.abs(on[30, 30].astype(int) - frame[30, 30]).max() <= 2
    assert 'bisenet_resnet_34' in Pipeline.required_models(Settings(face_shape=False, body_tone=0.5))
    p.reset_tracking()   # video path with temporal smoothing
    for _ in range(3):
        out = p.process(frame, src, Settings(body_tone=1.0), video=True)
    assert not np.array_equal(out[440, 320], frame[440, 320])


def test_landmarker_refines_kps():
    p = build()
    frame = scene()
    faces = p.analyse(frame, landmarks=False)
    raw = faces[0].kps.copy()
    Landmarker(Ident('lm')).refine(frame, faces[0])
    assert faces[0].landmarks68 is not None and not np.allclose(raw, faces[0].kps)


def test_rotated_detection():
    p = build()
    assert p.detector.detect_rotated(scene(body=False), 0.5)


def test_calibration_picks_right_convention():
    p = build()
    frame = scene()
    src = source_face(p)
    tgt = p.analyse(frame)[0]
    cfg = calibrate(p._sessions['alphaface_256'], None, frame, tgt, src, p.recogniser)
    assert cfg.mean == (0.5,) * 3 and cfg.std == (0.5,) * 3, cfg


def test_pipeline_all_features():
    p = build()
    frame = scene()
    src = source_face(p)
    for model in ('alphaface_256', 'hyperswap_1a_256'):
        for boost in (1, 2):
            out = p.process(frame, src, Settings(swapper=model, boost=boost, enhancer='gfpgan_1.4', occlusion=True,
                                                 expression=0.8, body_tone=0.8))
            assert out.shape == frame.shape and not np.array_equal(out, frame)
            assert np.array_equal(out[:20, :20], frame[:20, :20])
    assert np.array_equal(p.process(frame, src, Settings(face_mode='selected')), frame)


def test_expression_changes_result_and_is_optional():
    p = build()
    src = source_face(p)
    frame = scene(mouth=250)
    assert not np.array_equal(p.process(frame, src, Settings(expression=0.0)),
                              p.process(frame, src, Settings(expression=1.0)))
    assert 'live_portrait_generator' not in Pipeline.required_models(Settings(expression=0.0))


def test_video_tracking_recovers_lost_face():
    p = build()
    src = source_face(p)
    frame = scene()
    p.reset_tracking()
    p.process(frame, src, Settings(), video=True)
    real = p._sessions['retinaface_10g']
    calls = {'n': 0}
    class Flaky:
        def get_inputs(s): return real.get_inputs()
        def run(s, a, feed):
            calls['n'] += 1
            outs = real.run(a, feed)
            return [o * 0 for o in outs] if calls['n'] == 1 else outs
    p._sessions['retinaface_10g'] = Flaky()
    assert not np.array_equal(p.process(frame, src, Settings(expression=0.8), video=True), frame)


# ------------------------------------------------------------------ media --
def _ff(*args):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True)


def _streams(path):
    return subprocess.run(['ffmpeg', '-hide_banner', '-i', path], capture_output=True).stderr.decode()


def _swap_file(src, dst):
    info = media.probe(src)
    w = media.VideoWriter(dst, src, info)
    n = 0
    for f in media.iter_frames(src, info):
        w.write(255 - f); n += 1
    return info, n, w.close()


def test_video_keeps_sound_aac():
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'in.mp4'); dst = os.path.join(d, 'out ünïcode.mp4')
    _ff('-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=25', '-f', 'lavfi', '-i', 'sine=f=440',
        '-t', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', src)
    info, n, audio = _swap_file(src, dst)
    assert info.has_audio and n == 50 and audio == 'audio'
    assert 'Audio' in _streams(dst) and not os.path.exists(dst.replace('.mp4', '.swapframe-tmp.mp4'))


def test_video_keeps_sound_when_copy_impossible():
    """PCM audio in MKV can't be copied into MP4 -> must fall back to AAC, not drop the sound."""
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'in.mkv'); dst = os.path.join(d, 'out.mp4')
    _ff('-f', 'lavfi', '-i', 'testsrc=size=322x242:rate=30', '-f', 'lavfi', '-i', 'sine=f=300',
        '-t', '1.5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', '-shortest', src)
    info, n, audio = _swap_file(src, dst)
    assert audio == 'audio', audio
    s = _streams(dst)
    assert 'aac' in s and 'Audio' in s


def test_video_without_sound():
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'in.mp4'); dst = os.path.join(d, 'out.mp4')
    _ff('-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=25', '-t', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', src)
    info, n, audio = _swap_file(src, dst)
    assert not info.has_audio and audio == 'no-audio' and os.path.exists(dst)


def test_gif_in_and_out():
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'in.gif')
    _ff('-f', 'lavfi', '-i', 'testsrc=size=161x121:rate=10', '-t', '2', src)   # odd size on purpose
    info = media.probe(src)
    assert info and info.frames == 20 and (info.width, info.height) == (161, 121), info
    assert media.grab_frame(src, 1.0) is not None and media.grab_frame(src, 99) is not None
    gif_out = os.path.join(d, 'out.gif')
    _, n, audio = _swap_file(src, gif_out)
    assert n == 20 and audio == 'no-audio'
    back = media.probe(gif_out)
    assert back and (back.width, back.height) == (161, 121) and back.frames == 20, back
    mp4_out = os.path.join(d, 'out.mp4')
    _, _, audio = _swap_file(src, mp4_out)
    assert audio == 'no-audio' and media.probe(mp4_out).width == 160
    _, _, _ = _swap_file(os.path.join(d, 'out.gif'), os.path.join(d, 'again.gif'))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
