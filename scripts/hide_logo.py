"""Hides the printed logo (and the product code printed next to it) on every board of a panel photo.

Mark the logo once, on any photo of that board, and the script finds all the other copies on the
panel by template matching and pixelates them:

    python scripts/hide_logo.py --template panel.png --box 462 845 520 1145 --input panel.png --output panel_clean.png

--box is x0 y0 x1 y1 of one logo in the template photo. --left/--pad widen the hidden area around each
match (the product code is usually printed beside the logo). Always open the output and check it.
"""

import argparse

import cv2


def pixelate(img, x0, y0, x1, y1, cell=24):
    block = img[y0:y1, x0:x1]
    small = cv2.resize(block, (max(1, (x1 - x0) // cell), max(1, (y1 - y0) // cell)), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--box", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--left", type=int, default=240, help="extra pixels hidden to the left of each logo")
    ap.add_argument("--pad", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=0.45)
    a = ap.parse_args()

    x0, y0, x1, y1 = a.box
    tpl = cv2.imread(a.template, cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
    img = cv2.imread(a.input)
    f = 0.5
    res = cv2.matchTemplate(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=f, fy=f),
                            cv2.resize(tpl, None, fx=f, fy=f), cv2.TM_CCOEFF_NORMED)
    th, tw = tpl.shape
    found = 0
    while True:
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < a.min_score:
            break
        x, y = int(loc[0] / f), int(loc[1] / f)
        pixelate(img, max(0, x - a.left), max(0, y - a.pad), min(img.shape[1], x + tw + a.pad),
                 min(img.shape[0], y + th + a.pad))
        cv2.rectangle(res, (int(loc[0] - tw * f / 2), int(loc[1] - th * f / 2)),
                      (int(loc[0] + tw * f / 2), int(loc[1] + th * f / 2)), -1, -1)
        found += 1
    cv2.imwrite(a.output, img)
    print(f"{found} logo(s) hidden -> {a.output}. Open it and check before using it anywhere.")


if __name__ == "__main__":
    main()
