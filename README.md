# Swapframe

Clean, simple desktop face swapper for photos, videos and GIFs (Windows + Linux).

## Models
- **AlphaFace** (default) – newest swapper, built to stay clean on strongly turned heads
- **HyperSwap 1A / 1B / 1C** – very sharp on front-facing shots
- **InSwapper 128** – classic; pair with GPEN 512 at ~70–80%

## Expression
**Keep target expression** (on by default) uses LivePortrait to copy the original smile, eyes,
brows and mouth back onto the swapped face. 80% is a good start; go higher for big expressions,
lower if the face starts to look warped.

## Skin tone
- **Match body skin to new face** (off by default) – recolours the target's visible skin (neck, ears,
  arms, hands) to the source face's skin tone, so the face and body match. Only pixels that look like
  that person's skin are touched; it's smoothed over time in videos so it never flickers.
- **Match face to scene lighting** – adapts the new face to the scene's lighting.

## Natural look
- **Follow face shape** – face-parsing mask blends along the real outline
- **Precise alignment** – 68-point landmarks for turned heads and big expressions
- **Keep hands and hair in front** – occlusion mask
- **Steady video tracking** – smooths jitter and re-finds faces lost mid-turn

## Videos and GIFs
- Video output keeps the original sound (copied when it's AAC/MP3, otherwise converted to AAC so
  every player can play it). The status line tells you if the original had no sound.
- GIFs open like videos and can be saved as a looping GIF (high-quality palette) or MP4.

## Themes
System, Light, Dark, Midnight, Forest, Graphite, Rose, Sand and Mint – pick one from the top of the sidebar.
Drag the gap between the sidebar and the preview to resize the sidebar.

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
The release is tagged from `__version__` in `swapframe/__init__.py` (e.g. `v1.3.0`); if that version
was already released, it's published as `v1.3.0-build.<run number>`.

## Tips for natural results
- Give 2–5 sharp source photos from a few angles, with a neutral or relaxed expression.
- Use **High** detail; use **Max** only for close-ups.
- Leave Enhance off unless the target is blurry (it can soften expressions).

## Responsible use
Only swap faces of people who agreed to it. Several models are non-commercial / research only
(AlphaFace's paper is CC BY-NC-SA), so check licences before any commercial use.
