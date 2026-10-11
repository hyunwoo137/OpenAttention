# Demo: dense attention vs OpenAttention, side by side

Records Llama-3.1-8B-Instruct generating from a real 30,012-token document twice — once with
Hugging Face's out-of-the-box dense attention (default SDPA backend, eager) and once with the
deployed OpenAttention path at a 1% read — then renders both runs side by side in real time.

```bash
python demo/record_demo.py --method dense --device cuda:0 --out demo/runs/dense.json
python demo/record_demo.py --method ours  --device cuda:0 --out demo/runs/ours.json
python demo/render_demo.py --dense demo/runs/dense.json --ours demo/runs/ours.json \
    --mp4 docs/static/demo.mp4 --gif assets/demo.gif
```

Notes:
- The recorder streams tokens the way a server would (one host sync per generated token) and logs a
  wall-clock timestamp per token; nothing in the video is sped up (`--speed` changes playback only).
- Rendering needs `pip install imageio-ffmpeg pillow` (bundles a static ffmpeg; not required for
  anything else in the repo).
- `--question` changes the instruction appended after the document; the document is RULER qa_1
  request #0.
