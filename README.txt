
# Final Assignment: Object Tracking using SAM2

## Problem Statement

Track a known object (can of chowder) from one image to another using the Segment Anything Model 2 (SAM2). Given a ground-truth mask in the first image, extract the object’s bounding box, and track the same object in the second image using SAM2’s video tracking module.

## Steps

1. **Extract Bounding Box**: Use the mask from the first image to find object location.
2. **Object Tracking**: Use SAM2 to track the object in the second image using the bounding box.
3. **Visualization**: Show the original and tracked object with bounding boxes and masks.

## Requirements

- Python ≥ 3.8
- PyTorch + CUDA
- SAM2 repo installed from: https://github.com/facebookresearch/segment-anything-2
- Files needed:
  - `can_chowder_000001.jpg`
  - `can_chowder_000001_1_gt.png`
  - `can_chowder_000002.jpg`

## Run Instructions

```bash
python sam2_tracking_assignment.py
```

Make sure model weights (`sam2_hiera_tiny.pt`) and config (`sam2_hiera_t.yaml`) are in the same directory.

