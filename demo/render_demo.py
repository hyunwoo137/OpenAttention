#!/usr/bin/env python3
"""Render the side-by-side demo video from two record_demo.py runs (real-time playback).

    python demo/render_demo.py --dense demo/runs/dense.json --ours demo/runs/ours.json \
        --mp4 docs/static/demo.mp4 --gif assets/demo.gif
"""
import argparse
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG, PANE, EDGE = (13, 15, 20), (20, 24, 32), (45, 52, 66)
FG, DIM, ACC, ACC2 = (222, 226, 232), (130, 138, 152), (94, 200, 145), (255, 180, 84)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANSB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(draw, text, fnt, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            t = (cur + " " + word).strip()
            if draw.textlength(t, font=fnt) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def pane(draw, x, y, w, h, run, t, fonts, color):
    f_h, f_m, f_s = fonts
    draw.rounded_rectangle([x, y, x + w, y + h], 12, fill=PANE, outline=EDGE, width=2)
    draw.text((x + 18, y + 14), run["label"], font=f_h, fill=color)
    draw.text((x + 18, y + 44), f"{run['prompt_tokens']:,}-token document · Llama-3.1-8B · greedy",
              font=f_s, fill=DIM)
    ty = y + 78
    if t < run["prefill_s"]:
        dots = "." * (1 + int(t * 4) % 3)
        draw.text((x + 18, ty), f"prefilling the document{dots}", font=f_m, fill=DIM)
        n, ms = 0, None
    else:
        td = t - run["prefill_s"]
        n = sum(1 for s in run["stamps_s"] if s <= td)
        n = max(n, 1)
        text = run["pieces"][n - 1] + ("▌" if n < len(run["pieces"]) else "")
        lines = wrap(draw, text, f_m, w - 36)
        maxl = (y + h - 56 - ty) // 22
        for i, ln in enumerate(lines[-maxl:]):
            draw.text((x + 18, ty + 22 * i), ln, font=f_m, fill=FG)
        ms = 1e3 * run["stamps_s"][n - 1] / max(n - 1, 1)
    draw.line([x + 14, y + h - 44, x + w - 14, y + h - 44], fill=EDGE, width=1)
    stat = f"{n:3d} tokens   {('%5.1f ms/token' % ms) if ms else '   —'}   {t:5.1f} s elapsed"
    draw.text((x + 18, y + h - 34), stat, font=f_m, fill=color)


def endcard(draw, fonts, stats):
    f_title, f_big, f_m = fonts
    draw.text((W // 2, 92), "OpenAttention — 1% of the KV cache is enough", font=f_title,
              fill=FG, anchor="mm")
    xs, y = [W // 4, W // 2, 3 * W // 4], 240
    for (big, sub), xc in zip(stats, xs):
        draw.text((xc, y), big, font=f_big, fill=ACC, anchor="mm")
        for i, s in enumerate(sub):
            draw.text((xc, y + 58 + 24 * i), s, font=f_m, fill=DIM, anchor="mm")
    draw.text((W // 2, H - 150), "accuracy: RULER-32K HARD, 7 tasks × 200 requests · "
              "latency: one H100, 30K-token prompt, batch 1", font=f_m, fill=DIM, anchor="mm")
    draw.text((W // 2, H - 110), "vs. Hugging Face dense attention (default SDPA backend), same model, same prompt",
              font=f_m, fill=DIM, anchor="mm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense", required=True)
    p.add_argument("--ours", required=True)
    p.add_argument("--mp4", default="")
    p.add_argument("--gif", default="")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    a = p.parse_args()
    dense = json.load(open(a.dense))
    ours = json.load(open(a.ours))
    f_title = font(SANSB, 34)
    f_head = font(SANSB, 21)
    f_mono = font(MONO, 15)
    f_small = font(SANS, 14)
    f_big = font(SANSB, 52)
    T = max(dense["prefill_s"] + dense["stamps_s"][-1], ours["prefill_s"] + ours["stamps_s"][-1]) + 0.8
    sp = max(dense["tpot_ms"] / ours["tpot_ms"], 1.0)
    e2e = max(dense["e2e_s"] / ours["e2e_s"], 1.0)
    stats = [(f"{sp:.1f}×", ["faster decoding", f"{dense['tpot_ms']:.1f} → {ours['tpot_ms']:.1f} ms/token"]),
             (f"{e2e:.1f}×", ["faster end-to-end", f"{dense['e2e_s']:.1f} → {ours['e2e_s']:.1f} s"]),
             ("99%", ["of dense accuracy", "86.3 vs 87.0 at a 1% read"])]
    frames = []
    nplay = int(T / a.speed * FPS)
    for i in range(nplay + int(5.0 * FPS)):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        if i < nplay:
            t = i / FPS * a.speed
            d.text((W // 2, 34), "Generating from a 30K-token document — same GPU, same model",
                   font=f_title, fill=FG, anchor="mm")
            pw = (W - 3 * 20) // 2
            pane(d, 20, 70, pw, H - 95, dense, t, (f_head, f_mono, f_small), ACC2)
            pane(d, 40 + pw, 70, pw, H - 95, ours, t, (f_head, f_mono, f_small), ACC)
        else:
            endcard(d, (f_title, f_big, f_small), stats)
        frames.append(img)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s")
    if a.mp4:
        os.makedirs(os.path.dirname(a.mp4) or ".", exist_ok=True)
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen(
            [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", a.mp4],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for fr in frames:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        proc.wait()
        print("wrote", a.mp4, os.path.getsize(a.mp4) // 1024, "KB")
    if a.gif:
        os.makedirs(os.path.dirname(a.gif) or ".", exist_ok=True)
        small = [f.resize((880, 495)) for f in frames[:: FPS // 10]]           # 10 fps gif
        small[0].save(a.gif, save_all=True, append_images=small[1:], duration=100, loop=0, optimize=True)
        print("wrote", a.gif, os.path.getsize(a.gif) // 1024, "KB")


if __name__ == "__main__":
    main()
