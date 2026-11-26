# Swapframe

Clean, simple desktop face swapper for photos and videos (Windows + Linux).

## Models
- **AlphaFace** (default) – newest swapper, built to stay clean on strongly turned heads
- **HyperSwap 1A / 1B / 1C** – very sharp on front-facing shots
- **InSwapper 128** – classic; pair with GPEN 512 at ~70–80%

## Expression
**Keep target expression** (on by default) uses LivePortrait to copy the original smile, eyes,
brows and mouth back onto the swapped face. The slider sets how strongly; 80% is a good start.
Go higher for big expressions (laughing, shouting), lower if the face starts to look warped.

## Natural look
- **Follow face shape** – face-parsing mask blends along the real outline
- **Match skin tone and lighting** – adapts the new face to the scene
- **Precise alignment** – 68-point landmarks for turned heads and big expressions
- **Keep hands and hair in front** – occlusion mask
- **Steady video tracking** – smooths jitter and re-finds faces lost mid-turn
- Sideways / upside-down photos are detected automatically

AlphaFace's input format is detected automatically on first use and remembered in
`calibration.json` in the models folder. Delete that file to redo it.

Models download on first use (checksum-verified) into:
- Windows `%LOCALAPPDATA%\Swapframe\models`
- Linux `~/.local/share/swapframe/models`

## Run from source
```
pip install -r requirements.txt
python app.py
```

## Build + release
Every push to `main` builds Windows + Linux and publishes a GitHub Release automatically.
The release is tagged from `__version__` in `swapframe/__init__.py` (e.g. `v1.2.0`); if that version
was already released, it's published as `v1.2.0-build.<run number>`. Bump `__version__` for a clean tag.

## Tips for natural results
- Give 2–5 sharp source photos from a few angles, with a **neutral or relaxed** expression.
- Use **High** detail; use **Max** only for close-ups.
- Leave Enhance off unless the target is blurry (it can soften expressions).

## Responsible use
Only swap faces of people who agreed to it. Several models are non-commercial / research only
(AlphaFace's paper is CC BY-NC-SA), so check licences before any commercial use.
