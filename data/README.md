# Data layout

The panel photos used in this work are not distributed. To run the pipeline on your own boards, organize them like this:

```
data/<board>/
├── rois.json            ROI size and top-left corner of each ROI, in panel pixels
├── train/               good panel photos (≥ 5; ≈ 80 recommended), all aligned to the same reference
└── test/                optional
    ├── good/            good panels
    └── defect/          panels with defects
```

`rois.json`:

```json
{
  "roi_size": 512,
  "rois": {
    "roi_001": [812, 1440],
    "roi_002": [1404, 1440]
  }
}
```

Tips:

- Every photo must come from the same camera position and lighting. If the panel moves between shots, align the photos to a reference first (for example with `cv2.phaseCorrelate`); a shift of a few pixels is enough to raise the score of a good ROI.
- Variety beats quantity: 50 different panels are better than 200 photos of the same panel.
- Keep defective panels out of `train/`. The leave-one-out threshold tolerates a few that slip in (up to 10%), but each one makes that defect look "normal".
