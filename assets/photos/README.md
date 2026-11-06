# Real images

All photos here come from the production line (Board D), with the logo and product code pixelated by `scripts/hide_logo.py`.

| File | Used in | Status |
|---|---|---|
| `prod_result.jpg` | top of both READMEs | done (the station's result screen on a real inspection; product renamed to "Placa D", sidebar cropped out) |
| `panel_overview.jpg` / `panel_overview_pt.jpg` | Figure 1 | done |
| `panel_rois.jpg` | Figure 2 | done |
| `roi_mosaic.jpg` | Figure 3 | done |
| `joint_closeup.jpg` | Figure 4 (right side of the lighting animation) | done (a close-up of real joints, no logo in the frame) |
| `infer_output.jpg` | Figure 17 | done (a real inspection: job 1166) |

To add new photos (for example a panel with a real bridge, for a side-by-side of an intentional and an accidental short), hide the logos first and build the figure from the clean panel:

```bash
python scripts/hide_logo.py --template <panel.png> --box <x0 y0 x1 y1 of one logo> --input <panel.png> --output <panel_clean.png>
python scripts/make_photo_figures.py --board data/board_d --models models/board_d \
    --panel <panel_clean.png> --roi <roi_with_the_defect> \
    --intentional <panel>.png:<roi_with_a_bridge_by_design> --accidental <panel>.png:<roi_with_the_defect>
```

Open every image before committing and check that no logo, product code or serial label is visible.
