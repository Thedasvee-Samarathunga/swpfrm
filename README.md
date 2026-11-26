# Swapframe

Clean, simple desktop face swapper for photos and videos (Windows + Linux).

## Models
- **AlphaFace** (default) – newest swapper, built to stay clean on strongly turned heads
- **HyperSwap 1A / 1B / 1C** – very sharp on front-facing shots
- **InSwapper 128** – classic; pair with GPEN 512 at ~70–80%

## Natural look
- **Follow face shape** – face-parsing mask blends along the real outline (no box edges on profiles)
- **Match skin tone and lighting** – adapts the new face to the scene
- **Precise alignment** – 68-point landmarks for turned heads and big expressions
- **Keep hands and hair in front** – occlusion mask
- **Steady video tracking** – smooths jitter and re-finds faces the detector loses mid-turn
- Sideways / upside-down photos are detected automatically

AlphaFace's input format is detected automatically the first time you use it (a few seconds, once)
and remembered in `calibration.json` in the models folder. Delete that file to redo it.

Models download on first use (checksum-verified) into:
- Windows `%LOCALAPPDATA%\Swapframe\models`
- Linux `~/.local/share/swapframe/models`

## Run from source
```
pip install -r requirements.txt
python app.py
```

## Build
Every push to `main` builds Windows + Linux and publishes a GitHub Release automatically.
The release is tagged from `__version__` in `swapframe/__init__.py` (e.g. `v1.1.0`); if that version
was already released, it's published as `v1.1.0-build.<run number>`. Bump `__version__` for a clean tag.

## Tips for natural results
- Give 2–5 sharp source photos from a few angles (front, ¾ left, ¾ right).
- Use **High** detail; use **Max** only for close-ups.
- Leave Enhance off with AlphaFace / HyperSwap unless the target is blurry.

## Responsible use
Only swap faces of people who agreed to it. Several models are non-commercial / research only
(AlphaFace's paper is CC BY-NC-SA), so check licences before any commercial use.
