#!/usr/bin/env python3
"""Render the one-minute method explainer (docs/static/explainer.mp4) with PIL + ffmpeg.

    python docs/make_explainer.py            # writes docs/static/explainer.mp4 and explainer_poster.png
"""
import math
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (250, 251, 253)
INK, SUB, LINE = (27, 31, 39), (91, 100, 114), (210, 216, 226)
TEAL, AMBER, RED, BLUE = (15, 118, 110), (180, 83, 9), (185, 50, 50), (37, 99, 190)
CLUST = [(15, 118, 110), (37, 99, 190), (180, 83, 9), (130, 80, 180), (190, 70, 110), (90, 140, 40)]
F = "/usr/share/fonts/truetype/dejavu/"
f_title = ImageFont.truetype(F + "DejaVuSans-Bold.ttf", 40)
f_h = ImageFont.truetype(F + "DejaVuSans-Bold.ttf", 28)
f_b = ImageFont.truetype(F + "DejaVuSans.ttf", 21)
f_s = ImageFont.truetype(F + "DejaVuSans.ttf", 17)
f_big = ImageFont.truetype(F + "DejaVuSans-Bold.ttf", 64)

import random
random.seed(7)
N = 64                                            # tokens in the cache strip
ASG = [random.randrange(6) for _ in range(N)]     # cluster of each token
TOPK = {5, 19, 33, 47, 58}                        # selector picks
HOT = {2, 3}                                      # high-error clusters (ErrorGate expands these)
EXPAND = {i for i in range(N) if ASG[i] in HOT and i % 3 == 0 and i not in TOPK}


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def mix(c1, c2, a):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def strip(d, y, hl=None, colors=None, gone=None, proxy=None, a=1.0, label=True):
    """the KV cache as a strip of tokens; hl = set drawn tall, colors = per-token cluster colors,
    gone = alpha-faded set, proxy = cluster -> x center of its proxy token"""
    x0, x1 = 90, W - 90
    w = (x1 - x0) / N
    for i in range(N):
        c = (colors or {}).get(i, (168, 176, 188))
        if gone and i in gone:
            c = mix(c, BG, 0.85)
        tall = hl and i in hl
        hh = 46 if tall else 30
        yy = y - (hh - 30) // 2
        col = mix(BG, c, a)
        d.rounded_rectangle([x0 + i * w + 1, yy, x0 + (i + 1) * w - 1, yy + hh], 3, fill=col)
    if label:
        d.text((x0, y + 40), "KV cache of one request (30K+ tokens)", font=f_s, fill=mix(BG, SUB, a))
    if proxy:
        for cl, px in proxy.items():
            d.rounded_rectangle([px - 12, y + 70, px + 12, y + 100], 5, fill=CLUST[cl], outline=INK)
            d.text((px, y + 112), "×n", font=f_s, fill=CLUST[cl], anchor="ma")


def header(d, kicker, title, a=1.0):
    d.text((W // 2, 64), kicker, font=f_s, fill=mix(BG, TEAL, a), anchor="mm")
    d.text((W // 2, 110), title, font=f_h, fill=mix(BG, INK, a), anchor="mm")


def caption(d, lines, a=1.0):
    for i, ln in enumerate(lines):
        d.text((W // 2, H - 110 + 30 * i), ln, font=f_b, fill=mix(BG, SUB, a), anchor="mm")


def frame_at(t):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 330
    if t < 9:                                                             # S1: problem
        a = ease(t / .8)
        header(d, "OPENATTENTION", "Long-context decoding is bound by KV-cache reads", a)
        strip(d, y, a=a)
        if t > 2:
            k = ease((t - 2) / 1.2)
            q = (W // 2, 210)
            d.ellipse([q[0] - 16, q[1] - 16, q[0] + 16, q[1] + 16], fill=mix(BG, INK, k))
            d.text((q[0] + 26, q[1]), "query", font=f_s, fill=mix(BG, SUB, k), anchor="lm")
            nl = int(N * k) if t < 5 else len(TOPK)
            tgt = range(nl) if t < 5 else sorted(TOPK)
            for i in tgt:
                x = 90 + (W - 180) / N * (i + .5)
                d.line([q, (x, y)], fill=mix(BG, TEAL if t >= 5 else (170, 178, 190), k), width=2)
        caption(d, ["Dense attention touches every token at every step."] if t < 5 else
                   ["Sparse attention reads ~1% ...", "but what happens to the other 99%?"])
    elif t < 21:                                                          # S2: discard vs sample
        s = t - 9
        header(d, "THE TAIL PROBLEM", "Two existing answers, two failure modes")
        half = s >= 6
        if not half:
            a = ease(s / .8)
            strip(d, y, hl=TOPK, colors={i: TEAL for i in TOPK}, gone=set(range(N)) - TOPK, a=a)
            d.text((W // 2, 230), "1. Discard the tail (top-k methods)", font=f_b, fill=mix(BG, INK, a), anchor="mm")
            caption(d, ["The softmax mass of 99% of the cache simply disappears —", "the output is biased, and aggregation tasks collapse."], a)
            d.text((W // 2, 450), "output bias ✗", font=f_h, fill=mix(BG, RED, ease((s - 2.5) / 1)), anchor="mm")
        else:
            a = ease((s - 6) / .8)
            samp = {8, 22, 40, 54}
            strip(d, y, hl=TOPK | samp, colors={**{i: TEAL for i in TOPK}, **{i: AMBER for i in samp}},
                  gone=set(range(N)) - TOPK - samp, a=a)
            d.text((W // 2, 230), "2. Sample the tail (MagicPIG, vAttention)", font=f_b, fill=mix(BG, INK, a), anchor="mm")
            caption(d, ["A few random tokens stand in for the rest. At low reads the tail still", "holds heavy hitters the selector missed — the estimate gets noisy exactly when it matters."], a)
            d.text((W // 2, 450), "sampling error ∝ 1/√(read)  ✗", font=f_h, fill=mix(BG, RED, ease((s - 8.5) / 1)), anchor="mm")
    elif t < 40:                                                          # S3: the method
        s = t - 21
        header(d, "OPENATTENTION", "Keep the whole tail — as count-aware cluster proxies")
        colors = {i: CLUST[ASG[i]] for i in range(N)}
        if s < 4:                                                         # QGC clustering
            a = ease(s / 1.2)
            cols = {i: mix((168, 176, 188), colors[i], a) for i in range(N)}
            strip(d, y, colors=cols)
            caption(d, ["QGC clusters the keys once per prefill, in a low-rank projection", "aligned with the directions the queries actually use (training-free)."])
            d.text((W // 2, 230), "① Query-Geometry Clustering", font=f_b, fill=INK, anchor="mm")
        elif s < 8:                                                       # QGS top-k
            a = ease((s - 4) / 1)
            strip(d, y, hl=TOPK, colors=colors)
            for i in sorted(TOPK):
                x = 90 + (W - 180) / N * (i + .5)
                d.line([(W // 2, 210), (x, y)], fill=mix(BG, INK, a), width=2)
            d.ellipse([W // 2 - 14, 196, W // 2 + 14, 224], fill=INK)
            caption(d, ["② Every step, QGS scans rank-32 int8 key codes", "in the same geometry and takes the top-k."])
            d.text((W // 2, 160), "② Query-Geometry Selector", font=f_b, fill=INK, anchor="mm")
        elif s < 13:                                                      # ErrorGate
            a = ease((s - 8) / 1)
            strip(d, y, hl=TOPK | (EXPAND if s > 10 else set()), colors=colors)
            bx = W // 2 - 240
            d.text((W // 2, 180), "③ ErrorGate: spend exact reads where the estimated error is largest", font=f_b, fill=INK, anchor="mm")
            for j, cl in enumerate(range(6)):
                err = [0.2, 0.35, 0.95, 0.8, 0.25, 0.3][cl]
                d.rounded_rectangle([bx + j * 80, 250 - 40 * err * a, bx + j * 80 + 46, 250], 4,
                                    fill=mix(BG, CLUST[cl], 0.9))
                if cl in HOT and s > 10:
                    d.text((bx + j * 80 + 23, 258), "+reads", font=f_s, fill=CLUST[cl], anchor="ma")
            caption(d, ["Each cluster caches count / mean / variance statistics; the budget is", "water-filled by estimated softmax-weight error — not by centroid score."])
        else:                                                             # proxies
            a = ease((s - 13) / 1)
            px = {cl: 180 + cl * 170 for cl in range(6)}
            strip(d, y, hl=TOPK | EXPAND, colors=colors, proxy={cl: px[cl] for cl in range(6)})
            d.text((W // 2, 200), "④ Everything still unread folds into its cluster's proxy", font=f_b, fill=INK, anchor="mm")
            caption(d, ["One token per cluster: mean key, mean value, logit  z̄ + log n.", "The softmax stays mass-preserving — no bias, no sampling noise."])
    elif t < 50:                                                          # S4: kernel
        s = t - 40
        a = ease(s / .8)
        header(d, "OPENATTENTION.FLASH", "Three streams, one fused online-softmax pass", a)
        labels = [("selected tokens", TEAL), ("expanded tokens", BLUE), ("cluster proxies (×n)", AMBER)]
        for j, (lab, col) in enumerate(labels):
            yy = 220 + j * 90
            d.rounded_rectangle([170, yy, 560, yy + 56], 10, outline=col, width=3)
            d.text((365, yy + 28), lab, font=f_b, fill=col, anchor="mm")
            prog = ease((s - .5 - j * .3) / 1)
            d.line([(560, yy + 28), (560 + 180 * prog, yy + 28 + (390 - yy - 28) * prog * 0 + (310 - yy - 28) * prog)], fill=col, width=3)
        d.rounded_rectangle([740, 270, 1120, 390], 14, fill=(20, 24, 32))
        d.text((930, 310), "one Triton kernel", font=f_b, fill=(222, 226, 232), anchor="mm")
        d.text((930, 348), "online softmax over all three", font=f_s, fill=(150, 158, 170), anchor="mm")
        caption(d, ["Attention stage at a 1% read, batch 32:  1.196 → 0.114 ms per layer  (10×).",
                    "Deployed with fused selection + one CUDA graph per decode step."])
    else:                                                                 # S5: results
        s = t - 50
        a = ease(s / .8)
        d.text((W // 2, 150), "One percent is enough", font=f_title, fill=mix(BG, INK, a), anchor="mm")
        rows = [("dense", "87.04"), ("MagicPIG", "10.53"), ("HashAttention", "48.22"),
                ("vAttention", "36.29"), ("OpenAttention", "86.30")]
        for j, (nm, v) in enumerate(rows):
            aa = ease((s - .4 - j * .35) / .8)
            yy = 230 + j * 62
            col = TEAL if nm == "OpenAttention" else (SUB if nm != "dense" else INK)
            d.text((W // 2 - 120, yy), nm, font=f_b if nm != "OpenAttention" else f_h, fill=mix(BG, col, aa), anchor="rm")
            bw = float(v) / 87.04 * 420
            d.rounded_rectangle([W // 2 - 90, yy - 14, W // 2 - 90 + bw * aa, yy + 14], 7, fill=mix(BG, col, aa))
            d.text((W // 2 - 80 + bw * aa + 10, yy), v, font=f_b, fill=mix(BG, col, aa), anchor="lm")
        caption(d, ["RULER-32K HARD, 7 tasks × 200 requests, 1% total read, Llama-3.1-8B.",
                    "Code: github.com/hyunwoo137/OpenAttention"], ease((s - 3) / 1))
    return img


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(out, exist_ok=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    T = 62
    proc = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
         f"{out}/explainer.mp4"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    poster = None
    for i in range(T * FPS):
        fr = frame_at(i / FPS)
        if abs(i / FPS - 23.0) < 1 / (2 * FPS):
            poster = fr
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    (poster or frame_at(23)).save(f"{out}/explainer_poster.png")
    print("wrote", f"{out}/explainer.mp4", os.path.getsize(f"{out}/explainer.mp4") // 1024, "KB")


if __name__ == "__main__":
    main()
