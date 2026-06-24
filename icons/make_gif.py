"""Render Version A of the Waruka icon as an animated GIF (for places that
can't use animated SVG). The red disc sweeps left->right laying the dashed
trail, holds, then sweeps back (clean loop, opaque square tile).
Run:  python make_gif.py
"""
import os, math
from PIL import Image, ImageDraw

try:
    LANCZOS = Image.Resampling.LANCZOS
    NODITHER = Image.Dither.NONE
except AttributeError:
    LANCZOS = Image.LANCZOS
    NODITHER = Image.NONE

ROOT = os.path.dirname(os.path.abspath(__file__))
GREEN = (26, 158, 75)
GOLD = (253, 185, 19)
RED = (206, 17, 38)
PTS = [(26, 32), (40, 71), (50, 46), (60, 71), (74, 32)]


def _segments(pts):
    segs, L = [], 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        l = math.hypot(b[0] - a[0], b[1] - a[1])
        segs.append((a, b, l)); L += l
    return segs, L


def point_at(pts, frac):
    segs, L = _segments(pts)
    target = max(0.0, min(1.0, frac)) * L
    acc = 0.0
    for idx, (a, b, l) in enumerate(segs):
        if acc + l >= target or idx == len(segs) - 1:
            t = (target - acc) / l if l else 0.0
            t = max(0.0, min(1.0, t))
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        acc += l
    return pts[-1]


def vertex_fracs(pts):
    segs, L = _segments(pts)
    fr, acc = [0.0], 0.0
    for _, _, l in segs:
        acc += l; fr.append(acc / L)
    return fr  # fraction at each vertex (len == len(pts))


def dash_upto(d, pts, width, fill, dash, gap, kk, frac):
    segs, L = _segments(pts)
    maxlen = frac * L
    w = max(1, round(width * kk))
    pat = [(dash, True), (gap, False)]
    pi, pl, on = 0, pat[0][0], pat[0][1]
    travelled = 0.0
    for a, b, l in segs:
        if l == 0:
            continue
        ux, uy = (b[0] - a[0]) / l, (b[1] - a[1]) / l
        dpos = 0.0
        while dpos < l - 1e-9:
            step = min(pl, l - dpos)
            if travelled + step > maxlen:
                step = maxlen - travelled
                if step <= 0:
                    return
            if on and step > 0:
                d.line([((a[0] + ux * dpos) * kk, (a[1] + uy * dpos) * kk),
                        ((a[0] + ux * (dpos + step)) * kk, (a[1] + uy * (dpos + step)) * kk)],
                       fill=fill, width=w)
            dpos += step; travelled += step; pl -= step
            if travelled >= maxlen:
                return
            if pl <= 1e-6:
                pi = (pi + 1) % 2
                pl, on = pat[pi][0], pat[pi][1]


VF = vertex_fracs(PTS)


def frame(p, size=128, ss=3):
    s = size * ss
    img = Image.new("RGB", (s, s), GREEN)
    d = ImageDraw.Draw(img)
    kk = s / 100.0
    # gold W (always full)
    sp = [(x * kk, y * kk) for x, y in PTS]
    w = max(1, round(11 * kk))
    d.line(sp, fill=GOLD, width=w, joint="curve")
    for x, y in sp:
        r = w / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=GOLD)
    # red dashed trail up to fraction p
    dash_upto(d, PTS, 1.4, RED, 3, 5, kk, p)
    # vertex dots that have been passed
    for i, (vx, vy) in enumerate(PTS[:-1]):
        if p >= VF[i] - 1e-6:
            cx, cy, r = vx * kk, vy * kk, 2.6 * kk
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
    # moving disc at fraction p
    dx, dy = point_at(PTS, p)
    cx, cy, r = dx * kk, dy * kk, 7 * kk
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
    return img.resize((size, size), LANCZOS)


def build(size=128, fwd=20, hold=5):
    frames = [frame(i / fwd, size) for i in range(fwd + 1)]   # draw L->R
    frames += [frame(1.0, size) for _ in range(hold)]          # hold complete
    frames += [frame(i / fwd, size) for i in range(fwd, -1, -1)]  # erase R->L
    frames += [frame(0.0, size) for _ in range(hold)]          # hold empty
    # one shared palette for flicker-free colours
    pal = frames[fwd].convert("P", palette=Image.ADAPTIVE, colors=64)
    q = [f.quantize(palette=pal, dither=NODITHER) for f in frames]
    out = os.path.join(ROOT, "waruka-anim-a.gif")
    q[0].save(out, save_all=True, append_images=q[1:], duration=55, loop=0,
              optimize=False, disposal=1)
    return out, len(frames)


if __name__ == "__main__":
    p, n = build(128)
    print("wrote", p, "(%d frames, %d bytes)" % (n, os.path.getsize(p)))
