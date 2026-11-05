# Real images

All photos here come from the production server (Board D), with the logo and product code pixelated by `scripts/hide_logo.py`.

| File | Used in | Status |
|---|---|---|
| `panel_overview.jpg` | Figure 1 | done |
| `panel_overview_pt.jpg` | Figura 1 (README em português) | done |
| `panel_rois.jpg` | Figure 2 | done |
| `roi_mosaic.jpg` | Figure 3 | done |
| `infer_output.jpg` | Figure 17 | done (a real inspection: job 1166) |
| `hero.png` | top of README | **needs a panel with a real solder defect** |
| `attempt2_intentional_short.png` | Figure 5 | **needs the ROI with a bridge by design and a photo where the same bridge is a defect** |

The server deletes the photos of each inspection after about two days, so the two missing ones have to come from a new inspection. Once a defective panel is inspected:

```bash
python scripts/hide_logo.py --template <panel.png> --box <x0 y0 x1 y1 of one logo> --input <panel.png> --output <panel_clean.png>
python scripts/make_photo_figures.py --board data/board_d --models models/board_d \
    --panel <panel_clean.png> --roi <roi_with_the_defect> \
    --intentional <panel>.png:<roi_with_a_bridge_by_design> --accidental <panel>.png:<roi_with_the_defect>
```

Open every image before committing and check that no logo, product code or serial label is visible.
