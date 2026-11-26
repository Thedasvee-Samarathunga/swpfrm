# Swapframe

Clean, simple desktop face swapper for photos and videos.

- **HyperSwap 1A / 1B / 1C** (FaceFusion, 256 px) plus **InSwapper 128**
- Pixel boost detail (Standard / High / Max) for sharper faces
- Optional GFPGAN 1.4 / GPEN 512 enhancement with strength control
- Occlusion mask keeps hands, hair and glasses in front of the new face
- Blend several source photos into one stronger identity
- Swap everyone, only the people you pick, or just the largest face
- Video: steady landmark tracking, live preview while rendering, original audio kept
- GPU: DirectML on Windows (NVIDIA / AMD / Intel), CoreML on macOS, CPU everywhere

Models download automatically on first use (verified with FaceFusion's checksums) into:
- Windows `%LOCALAPPDATA%\Swapframe\models`
- macOS `~/Library/Application Support/Swapframe/models`
- Linux `~/.local/share/swapframe/models`

## Run from source
```
pip install -r requirements.txt
python app.py
```

## Build
Push to `main` (or run the workflow manually) and grab the zips from the run's **Artifacts**.
Push a tag like `v1.0.0` to also publish a GitHub Release.

## Tips
- Use 2-5 sharp, well-lit photos of the source face from slightly different angles.
- HyperSwap 1B is often the sharpest; 1A is the most stable. Try 1A if 1B leaves a face unchanged.
- HyperSwap looks best with Enhance off. InSwapper looks best with GPEN 512 at around 70-80%.

## Responsible use
Only swap faces of people who agreed to it. The swap and enhancer models have their own licences
(several are non-commercial / research only), so check them before any commercial use.
