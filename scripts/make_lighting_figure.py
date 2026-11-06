"""Animation of the tiered RGB lighting: how three LED strips turn the slope of a solder joint into color.

    python scripts/make_lighting_figure.py                # assets/figures/lighting.gif
    FIG_LANG=pt python scripts/make_lighting_figure.py    # assets/figures/lighting_pt.gif

Geometry (cross-section, camera looking straight down): solder is shiny, so each point reflects
toward the camera only the light that arrives at twice its tilt from the vertical (mirror law).
The red strip sits next to the camera, green in the middle, blue low, close to the board:
flat solder reflects red, a moderate slope green, a steep slope blue, and anything steeper than
the lowest strip stays dark. The matte solder mask scatters light in every direction and looks dim.
Technique: tiered-color illumination (Capson and Eng, IEEE TPAMI 1988).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402  (palette, rcParams, language helpers)

plt = mf.plt
T, out = mf.T, mf.out
ROOT = HERE.parent

RED, GREEN, BLUE = "#e8412c", "#2fbf55", "#2f6fe8"
DARK, MASK, COPPER, SOLDER, PIN = "#22262b", "#2c4a35", "#b87333", "#c9cdd2", "#8d939a"
# strips, as angle from the vertical seen from the joint (degrees): red next to the camera, blue low
STRIPS = [(RED, 4, 24), (GREEN, 24, 50), (BLUE, 50, 84)]
R_LED = 9.0
K = 1.75          # size of the joint in the drawing (the slopes, and so the colors, do not depend on it)
PAD = 3.4 * K     # half-width of the pad


def profile(x):
    """Height of the surface: pin top (flat), concave fillet, flat edge of the pad, then the board."""
    return K * _base(np.asarray(x) / K)


def _base(x):
    ax = np.abs(x)
    y = np.zeros_like(x)
    pin = ax <= 0.45
    fil = (ax > 0.45) & (ax <= 2.9)
    pad = (ax > 2.9) & (ax <= 3.4)
    y[pin] = 1.75
    y[fil] = 0.12 + 1.45 * ((2.9 - ax[fil]) / 2.45) ** 2
    y[pad] = 0.12
    return y


def shade(x):
    """Color seen by the camera at x, and the direction (unit vector) the reflected light came from."""
    if abs(x) > PAD:
        return "mask", None
    h = 1e-3
    dy = (profile(np.array([x + h]))[0] - profile(np.array([x - h]))[0]) / (2 * h)
    n = np.array([-dy, 1.0]) / np.hypot(dy, 1.0)
    v = np.array([0.0, 1.0])  # toward the camera
    src = 2 * np.dot(n, v) * n - v  # mirror: where the light must come from
    phi = np.degrees(np.arccos(np.clip(src[1], -1, 1)))
    for color, a, b in STRIPS:
        if a - 4 <= phi < b:  # the red band reaches the lens axis (ring around the camera)
            return color, src
    return DARK, src


def main():
    xs = np.linspace(-PAD - 0.8, PAD + 0.8, 400)
    ys = profile(xs)
    seen = [shade(x)[0] for x in xs]
    path = np.linspace(xs[0], xs[-1], 70)
    stretch = 9.6 / xs[-1]  # the "seen from above" strip uses the whole width

    fig = plt.figure(figsize=(12, 5.0))
    ax = fig.add_axes((0.01, 0.02, 0.62, 0.84))
    ax2 = fig.add_axes((0.655, 0.16, 0.33, 0.66))
    mf.title(fig, T("Three LED strips turn the slope of the solder into color",
                    "Três fitas de LED transformam a inclinação da solda em cor"),
             T("the camera only gets, from each point of the solder, the light of the strip that sits at the right angle",
               "de cada ponto da solda, a câmera só recebe a luz da fita que está no ângulo certo"))
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-2.4, 11.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # LED strips (both sides), drawn as arcs around the joint
    for color, a, b in STRIPS:
        for s in (-1, 1):
            t = np.radians(np.linspace(a + 1, b - 1, 30))
            ax.plot(s * R_LED * np.sin(t), R_LED * np.cos(t), color=color, lw=9, solid_capstyle="butt")
    names = [T("red strip", "fita vermelha"), T("green strip", "fita verde"), T("blue strip", "fita azul")]
    for (color, a, b), name in zip(STRIPS, names):
        m = np.radians((a + b) / 2)
        ax.text((R_LED + 0.7) * np.sin(m), (R_LED + 0.7) * np.cos(m), name, color=color, fontsize=9,
                va="center", ha="left", fontweight="bold")

    # camera
    ax.add_patch(plt.Rectangle((-0.9, 10.1), 1.8, 1.0, fc=mf.INK2, ec="none"))
    ax.add_patch(plt.Polygon([[-0.45, 10.1], [0.45, 10.1], [0.3, 9.7], [-0.3, 9.7]], fc=mf.INK2, ec="none"))
    ax.text(-1.15, 10.6, T("camera", "câmera"), ha="right", va="center", fontsize=9, color=mf.INK2)

    # board, pad, pin and solder
    ax.add_patch(plt.Rectangle((-10, -0.7), 20, 0.7, fc=MASK, ec="none"))
    ax.add_patch(plt.Rectangle((-PAD, 0.0), 2 * PAD, 0.12 * K, fc=COPPER, ec="none"))
    ax.fill_between(xs, 0.0, ys, where=np.abs(xs) <= PAD, color=SOLDER, lw=0)
    ax.text(8.3, -0.35, T("board (matte mask)", "placa (máscara fosca)"), ha="center", va="center",
            fontsize=8, color="#d8dcd9")
    ax.text(0, 1.75 * K + 0.35, T("solder joint (cut)", "solda (em corte)"), ha="center", fontsize=9, color=mf.INK2)

    # "what the camera sees" strip under the board
    ax.text(-10, -1.25, T("what the camera sees from above:", "o que a câmera vê, de cima:"), fontsize=8.5,
            color=mf.INK2, va="center")
    strip_y, strip_h = -2.25, 0.7
    seen_art = ax.scatter([], [], s=1)
    seen_rects = []
    for i in range(len(xs) - 1):
        c = seen[i]
        rc = plt.Rectangle((xs[i] * stretch, strip_y), (xs[i + 1] - xs[i]) * stretch + 0.004, strip_h,
                           fc="#3a4a40" if c == "mask" else c, ec="none", visible=False)
        ax.add_patch(rc)
        seen_rects.append(rc)
    surf_line, = ax.plot([], [], lw=5, solid_capstyle="butt")
    surf_segments = []
    for i in range(len(xs) - 1):
        c = seen[i]
        seg, = ax.plot(xs[i:i + 2], ys[i:i + 2], lw=4.5, solid_capstyle="butt",
                       color="#3a4a40" if c == "mask" else c, visible=False)
        surf_segments.append(seg)
    cam_ray, = ax.plot([], [], color=mf.MUTED, lw=1.2, ls=(0, (3, 2)))
    led_ray, = ax.plot([], [], lw=2.2)
    dot, = ax.plot([], [], "o", ms=7, mec="white", mew=1.2)
    label = ax.text(0, 5.6, "", ha="center", fontsize=11, fontweight="bold")

    word = {RED: T("flat → red", "plano → vermelho"), GREEN: T("tilted → green", "inclinado → verde"),
            BLUE: T("steep → blue", "bem inclinado → azul"), DARK: T("too steep → dark", "inclinado demais → escuro"),
            "mask": T("matte mask → dim", "máscara fosca → escura")}

    # right: a real joint of Board D
    import cv2

    photo = cv2.cvtColor(cv2.imread(str(ROOT / "assets" / "photos" / "joint_closeup.jpg")), cv2.COLOR_BGR2RGB)
    ax2.imshow(photo)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title(T("Real joints of Board D, seen from above", "Soldas reais da Placa D, vistas de cima"),
                  fontsize=10, color=mf.INK, loc="left")
    ax2.text(0.0, -0.04, T("flat top in red, the slope down to the board in blue,\na thin green line in between",
                            "topo achatado em vermelho, a encosta até a placa em azul\ne um fio verde no meio"),
             transform=ax2.transAxes, fontsize=9, color=mf.INK2, va="top")

    hold = 22
    frames = len(path) + hold

    def update(k):
        x = path[min(k, len(path) - 1)]
        i_now = np.searchsorted(xs, x)
        for i, (rc, seg) in enumerate(zip(seen_rects, surf_segments)):
            vis = i < i_now
            rc.set_visible(vis)
            seg.set_visible(vis and abs(xs[i]) <= PAD)
        if k >= len(path):
            cam_ray.set_data([], [])
            led_ray.set_data([], [])
            dot.set_data([], [])
            label.set_text("")
            return []
        y = profile(np.array([x]))[0]
        c, src = shade(x)
        cam_ray.set_data([x, x], [y, 9.7])
        if src is not None and c != DARK:
            led_ray.set_data([x, x + src[0] * R_LED * 1.0], [y, y + src[1] * R_LED * 1.0])
            led_ray.set_color(c)
        else:
            led_ray.set_data([], [])
        col = "#3a4a40" if c == "mask" else c
        dot.set_data([x], [y])
        dot.set_color(col)
        label.set_text(word[c])
        label.set_color(mf.INK2 if c in ("mask", DARK) else c)
        return []

    from matplotlib.animation import FuncAnimation, PillowWriter

    anim = FuncAnimation(fig, update, frames=frames, interval=110)
    anim.save(out("lighting.gif"), writer=PillowWriter(fps=9), dpi=85)
    update(len(path) - 1)
    for i, (rc, seg) in enumerate(zip(seen_rects, surf_segments)):  # still: everything drawn
        rc.set_visible(True)
        seg.set_visible(abs(xs[i]) <= PAD)
    plt.close(fig)
    # shared 96-color palette: same look, a fraction of the size
    from PIL import Image, ImageSequence

    g = Image.open(out("lighting.gif"))
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(g)]
    durs = [f.info.get("duration", 110) for f in ImageSequence.Iterator(Image.open(out("lighting.gif")))]
    pal = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    small = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    small[0].save(out("lighting.gif"), save_all=True, append_images=small[1:], duration=durs, loop=0, optimize=True)
    print("saved", out("lighting.gif"), round(out("lighting.gif").stat().st_size / 2**20, 1), "MB")


if __name__ == "__main__":
    main()
