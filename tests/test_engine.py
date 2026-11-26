"""Offline tests: exercise the whole pipeline with fake ONNX sessions."""
import os, sys, tempfile
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swapframe import engine, media
from swapframe.engine import Pipeline, Settings, implode_pixel_boost, explode_pixel_boost, paste_back, box_mask

class IO:
    def __init__(s, name, shape, type='tensor(float)'): s.name, s.shape, s.type = name, shape, type

class FakeDetector:
    """Emits one face at a known position in SCRFD output layout."""
    def __init__(s, box, kps): s.box, s.kps = box, kps
    def get_inputs(s): return [IO('input.1', [1, 3, '?', '?'])]
    def run(s, _, feed):
        size = feed['input.1'].shape[2]
        outs_s, outs_b, outs_k = [], [], []
        for stride in (8, 16, 32):
            n = (size // stride) ** 2 * 2
            sc = np.zeros((n, 1), np.float32); bb = np.zeros((n, 4), np.float32); kp = np.zeros((n, 10), np.float32)
            if stride == 16:
                cx, cy = (s.box[0] + s.box[2]) / 2, (s.box[1] + s.box[3]) / 2
                gx, gy = int(cx // 16), int(cy // 16)
                idx = (gy * (size // 16) + gx) * 2
                ax, ay = gx * 16, gy * 16
                sc[idx] = 0.95
                bb[idx] = [(ax - s.box[0]) / 16, (ay - s.box[1]) / 16, (s.box[2] - ax) / 16, (s.box[3] - ay) / 16]
                kp[idx] = ((s.kps - [ax, ay]) / 16).reshape(-1)
            outs_s.append(sc); outs_b.append(bb); outs_k.append(kp)
        return outs_s + outs_b + outs_k

class FakeRec:
    def get_inputs(s): return [IO('input', [1, 3, 112, 112])]
    def run(s, _, feed):
        x = feed['input']; v = np.resize(x.mean(axis=(2, 3)).reshape(-1), 512).astype(np.float32) + 1
        return [v[None]]

class FakeSwap:
    """HyperSwap-like: (source[1,512], target[1,3,256,256]) -> (output, mask). Returns inverted image."""
    def __init__(s): s.calls = 0
    def get_inputs(s): return [IO('source', [1, 512]), IO('target', [1, 3, 256, 256])]
    def get_outputs(s): return [IO('mask', [1, 1, 256, 256]), IO('output', [1, 3, 256, 256])]
    def run(s, names, feed):
        s.calls += 1
        assert names == ['output'], names
        assert feed['target'].shape == (1, 3, 256, 256) and feed['source'].shape == (1, 512)
        return [-feed['target']]

class FakeEnh:
    def get_inputs(s): return [IO('input', [1, 3, 512, 512])]
    def run(s, _, feed): return [feed['input']]

class FakeXseg:
    def get_inputs(s): return [IO('input', [1, 256, 256, 3])]
    def run(s, _, feed): return [np.ones((1, 256, 256, 1), np.float32)]

def test_pixel_boost_roundtrip():
    crop = np.random.rand(512, 512, 3).astype(np.float32)
    tiles = implode_pixel_boost(crop, 2, 256)
    assert tiles.shape == (4, 256, 256, 3)
    assert np.allclose(explode_pixel_boost(list(tiles), 2, 256), crop)

def test_paste_back_identity():
    frame = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    kps = np.array([[170, 130], [230, 130], [200, 165], [178, 200], [222, 200]], np.float32)
    crop, m = engine.warp_face(frame, kps, 'arcface_128', 256)
    out = paste_back(frame.copy(), crop.astype(np.float32), np.ones((256, 256), np.float32) * 0, m)
    assert np.array_equal(out, frame)

def build_pipeline(frame_kps, box):
    p = Pipeline('CPU')
    swap = FakeSwap()
    p._sessions = {'retinaface_10g': FakeDetector(box, frame_kps), 'arcface_w600k_r50': FakeRec(),
                   'hyperswap_1a_256': swap, 'gfpgan_1.4': FakeEnh(), 'xseg_1': FakeXseg()}
    return p, swap

def test_pipeline():
    frame = np.full((480, 640, 3), 128, np.uint8)
    cv2.circle(frame, (320, 240), 80, (40, 160, 220), -1)
    box = np.array([240, 150, 400, 340], np.float32)
    kps = np.array([[290, 220], [350, 220], [320, 255], [295, 290], [345, 290]], np.float32)
    p, swap = build_pipeline(kps, box)
    faces = p.analyse(frame)
    assert len(faces) == 1, faces
    assert np.allclose(faces[0].bbox, box, atol=1e-3) and np.allclose(faces[0].kps, kps, atol=1e-3)
    src = p.source_faces([frame])
    for boost, calls in ((1, 1), (2, 4)):
        swap.calls = 0
        out = p.process(frame, src, Settings(boost=boost, enhancer='gfpgan_1.4', occlusion=True))
        assert swap.calls == calls
        assert out.shape == frame.shape and not np.array_equal(out, frame)
        assert np.array_equal(out[:40], frame[:40])           # untouched far away from face
    # selected mode with no reference -> no change
    out = p.process(frame, src, Settings(face_mode='selected'))
    assert np.array_equal(out, frame)
    out = p.process(frame, src, Settings(face_mode='selected', reference=[src[0].normed]))
    assert not np.array_equal(out, frame)
    # video + stabilisation path
    p.reset_tracking()
    for _ in range(3):
        p.process(frame, src, Settings(), video=True)

def test_video_roundtrip():
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'in.mp4'); dst = os.path.join(d, 'out ünïcode.mp4')
    os.system(f'ffmpeg -v error -y -f lavfi -i testsrc=size=320x240:rate=25 -f lavfi -i sine=f=440 -t 2 '
              f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{src}"')
    info = media.probe(src)
    assert info and (info.width, info.height) == (320, 240) and abs(info.fps - 25) < 0.1, info
    w = media.VideoWriter(dst, src, info)
    n = 0
    for f in media.iter_frames(src, info):
        w.write(255 - f); n += 1
    w.close()
    assert n == 50, n
    out = media.probe(dst)
    assert out and out.frames >= 49
    import subprocess
    txt = subprocess.run(['ffmpeg', '-hide_banner', '-i', dst], capture_output=True).stderr.decode()
    assert 'Audio' in txt
    img = os.path.join(d, 'ïmg.png')
    media.write_image(img, media.grab_frame(src, 1.0))
    assert media.read_image(img).shape == (240, 320, 3)

if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
