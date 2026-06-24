"""Build Waruka icon assets: hybrid app .ico, favicons, PWA icons, monochrome.
Thin-dashed design at every size except 16px, which uses the clean W+disc fallback.
Run:  python build_assets.py
"""
import os, io, struct, math
from PIL import Image, ImageDraw

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
os.makedirs(WEB, exist_ok=True)

GREEN = (26, 158, 75, 255)   # #1A9E4B
GOLD  = (253, 185, 19, 255)  # #FDB913
RED   = (206, 17, 38, 255)   # #CE1126
WHITE = (255, 255, 255, 255)
BLACK = (20, 20, 22, 255)

THIN = dict(pts=[(26, 32), (40, 71), (50, 46), (60, 71), (74, 32)], w=11,
            rail=dict(w=1.4, dash=3, gap=5), dots_r=2.6, disc_r=7)
FALL = dict(pts=[(26, 33), (40, 70), (50, 47), (60, 70), (74, 33)], w=11, disc_r=7)


def _dash(d, pts, width, fill, dash, gap, kk):
    sp = [(p[0] * kk, p[1] * kk) for p in pts]
    w = max(1, round(width * kk))
    pat = [(dash * kk, True), (gap * kk, False)]
    pi, pl, on = 0, pat[0][0], pat[0][1]
    for i in range(len(sp) - 1):
        a, b = sp[i], sp[i + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L == 0:
            continue
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        dpos = 0.0
        while dpos < L - 1e-9:
            step = min(pl, L - dpos)
            if on:
                d.line([(a[0] + ux * dpos, a[1] + uy * dpos),
                        (a[0] + ux * (dpos + step), a[1] + uy * (dpos + step))], fill=fill, width=w)
            dpos += step
            pl -= step
            if pl <= 1e-6:
                pi = (pi + 1) % 2
                pl, on = pat[pi][0], pat[pi][1]


def glyph(size, cfg, gold=GOLD, red=RED, mono=None):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    kk = size / 100.0
    wc = mono or gold
    rc = mono or red
    sp = [(p[0] * kk, p[1] * kk) for p in cfg["pts"]]
    w = max(1, round(cfg["w"] * kk))
    d.line(sp, fill=wc, width=w, joint="curve")
    for x, y in sp:
        r = w / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=wc)
    if cfg.get("rail") and not mono:
        rr = cfg["rail"]
        _dash(d, cfg["pts"], rr["w"], rc, rr["dash"], rr["gap"], kk)
    if cfg.get("dots_r") and not mono:
        for p in cfg["pts"][:-1]:
            cx, cy = p[0] * kk, p[1] * kk
            r = cfg["dots_r"] * kk
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rc)
    cx, cy = cfg["pts"][-1][0] * kk, cfg["pts"][-1][1] * kk
    r = cfg["disc_r"] * kk
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rc)
    return img


def tile(size, rounded=True, color=GREEN):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if rounded:
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=round(24 * size / 100.0), fill=color)
    else:
        d.rectangle([0, 0, size - 1, size - 1], fill=color)
    return img


def app_icon(size, cfg, rounded=True):
    ss = size * 4
    base = tile(ss, rounded)
    base.alpha_composite(glyph(ss, cfg))
    return base.resize((size, size), LANCZOS)


def maskable(size, cfg, scale=0.72):
    ss = size * 4
    base = tile(ss, rounded=False)
    g = glyph(int(ss * scale), cfg)
    off = (ss - g.width) // 2
    base.alpha_composite(g, (off, off))
    return base.resize((size, size), LANCZOS)


def mono_icon(size, color=WHITE):
    ss = size * 4
    return glyph(ss, FALL, mono=color).resize((size, size), LANCZOS)


def write_ico(path, images):
    pngs = []
    for im in images:
        b = io.BytesIO()
        im.save(b, format="PNG")
        pngs.append(b.getvalue())
    n = len(images)
    out = struct.pack("<HHH", 0, 1, n)
    off = 6 + 16 * n
    for im, png in zip(images, pngs):
        w = 0 if im.width >= 256 else im.width
        h = 0 if im.height >= 256 else im.height
        out += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(png), off)
        off += len(png)
    out += b"".join(pngs)
    with open(path, "wb") as f:
        f.write(out)


def savepng(im, name, web=False):
    p = os.path.join(WEB if web else ROOT, name)
    im.save(p, format="PNG")


# 1) Main app icon: hybrid (16 = fallback, rest = thin-dashed)
write_ico(os.path.join(ROOT, "waruka.ico"),
          [app_icon(256, THIN), app_icon(128, THIN), app_icon(64, THIN),
           app_icon(48, THIN), app_icon(32, THIN), app_icon(16, FALL)])

# 2) Web favicons
write_ico(os.path.join(WEB, "favicon.ico"),
          [app_icon(48, THIN), app_icon(32, THIN), app_icon(16, FALL)])
savepng(app_icon(16, FALL), "favicon-16x16.png", web=True)
savepng(app_icon(32, THIN), "favicon-32x32.png", web=True)

# 3) Apple + Android (full square; the OS rounds)
savepng(app_icon(180, THIN, rounded=False), "apple-touch-icon.png", web=True)
savepng(app_icon(192, THIN, rounded=False), "android-chrome-192x192.png", web=True)
savepng(app_icon(512, THIN, rounded=False), "android-chrome-512x512.png", web=True)

# 4) Maskable (content inside the safe zone)
savepng(maskable(512, THIN), "maskable-512x512.png", web=True)

# 5) Monochrome (taskbar / tray) - white and black
write_ico(os.path.join(ROOT, "waruka-mono-white.ico"),
          [mono_icon(256, WHITE), mono_icon(64, WHITE), mono_icon(32, WHITE), mono_icon(16, WHITE)])
write_ico(os.path.join(ROOT, "waruka-mono-black.ico"),
          [mono_icon(256, BLACK), mono_icon(64, BLACK), mono_icon(32, BLACK), mono_icon(16, BLACK)])
savepng(mono_icon(32, WHITE), "waruka-mono-white-32.png")
savepng(mono_icon(32, BLACK), "waruka-mono-black-32.png")

# 6) Refreshed large preview
savepng(app_icon(256, THIN), "waruka-thin-dashed-256.png")

# 7) Web manifest
manifest = """{
  "name": "Waruka",
  "short_name": "Waruka",
  "icons": [
    { "src": "/android-chrome-192x192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/android-chrome-512x512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/maskable-512x512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "theme_color": "#1A9E4B",
  "background_color": "#1A9E4B",
  "display": "standalone"
}
"""
with open(os.path.join(WEB, "site.webmanifest"), "w", encoding="utf-8") as f:
    f.write(manifest)

print("WROTE assets under", ROOT)
for base in (ROOT, WEB):
    for fn in sorted(os.listdir(base)):
        fp = os.path.join(base, fn)
        if os.path.isfile(fp):
            print("  %8d  %s" % (os.path.getsize(fp), os.path.relpath(fp, ROOT)))
print("DONE")
