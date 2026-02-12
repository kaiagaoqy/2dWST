# 2D Wide-Angle Scene Tracing (2DWST) Analysis

A research project for analyzing hand-traced object boundaries in wide-angle scene images, with applications in visual perception studies for individuals with low vision.

## Overview

This project processes and analyzes hand-traced trajectories of objects in panoramic scene images. It uses advanced point cloud registration techniques (Coherent Point Drift - CPD) to match hand-traced boundaries with ground truth annotations, enabling quantitative evaluation of object detection and recognition performance.

## Project Structure

```
2dwst/
├── main.py                 # Main processing pipeline
├── process.py              # Trajectory processing and matching
├── stitching.py            # Panorama stitching and image registration
├── statistics.ipynb        # Statistical analysis notebook
├── utils/
│   └── registor.py         # CPD registration and matching algorithms
├── scenes/                 # Scene images at different viewing angles
│   ├── kidroom/
│   ├── livingroom/
│   ├── studyroom/
│   ├── workshop/
│   └── scene_dict.json     # Scene metadata and object catalog
├── annotations/            # Ground truth object boundary annotations
├── data/                   # Participant hand-tracing data (SubXXX/)
├── metadata/               # Panorama homography matrices and metadata
├── results/                # Processing results and visualizations
└── figures/                # Generated figure outputs
```

## Key Features

### 1. **Trajectory Processing**
- Converts raw hand-traced points into valid polygons
- Extracts outermost boundaries from complex trajectories
- Samples polygons at regular intervals for consistent comparison
- Handles multi-level nested trajectory data structures

### 2. **Point Cloud Registration (CPD)**
- **Rigid Registration**: Constrained rotation (±5°) and scaling (0.95-1.05x)
- **Deformable Registration**: Non-rigid refinement using Coherent Point Drift
- **Multi-phase Resampling**: Tests multiple alignment phases for optimal matching
- **Hybrid Matching**: Combines relaxed bidirectional best-search (BBS) with Hungarian algorithm

### 3. **Panorama Stitching**
- Registers multiple viewing angles to a reference view (-90°)
- Uses SIFT feature detection and FLANN-based matching
- Computes homography matrices for cross-view trajectory projection
- Supports subset stitching with anchor views and fallback matching

### 4. **Viewing Angle Analysis**
- Converts pixel coordinates to viewing angles (degrees)
- Computes angular distances between matched point pairs
- Accounts for camera intrinsic parameters

## Experimental Setup

### Scenes
Four indoor environments with varying complexity:
- **Living Room**: 20 objects (furniture, decorations, hazards)
- **Kid's Room**: 26 objects (toys, furniture, storage)
- **Study Room**: 18 objects (desk, plants, electronics)
- **Workshop**: 19 objects (tools, equipment, furniture)

### Viewing Conditions
- **Angles**: -50° to -130° (10° increments)
- **Lighting**: Normal and low-light conditions
- **Vision**: Normal vision (NV) and low vision (LV) participants

### Participants
- Multiple subjects (Sub002-Sub321)
- Vision assessment data (visual acuity, contrast sensitivity)
- Two visibility conditions per subject

## Data Format

### Input Data (`data/SubXXX/*.json`)
Each participant file contains:
```json
{
  "file": ["scene/condition/scene_angle.jpg", ...],
  "annotations": [
    {
      "detected": true,
      "detection_confidence": 5,
      "recognized": true,
      "recognition_confidence": 5,
      "recognized_name": "table",
      "hazard": false,
      "gt": ["scene_001", "Table"],
      "judgment": "A"
    }
  ],
  "points": [[[x, y, z], ...], ...],
  "camera": {...}
}
```

### Output Data (`results/SubXXX/*.json`)
Processed files include additional `match` field:
```json
{
  "match": [
    {
      "ind": 0,
      "matched_label": "table",
      "avg_distance": 1.186,
      "distances": [1.009, 0.990, ...],
      "final_matches": [[[traced_x, traced_y], [gt_x, gt_y]], ...],
      "draw_traj": [[x1, y1], [x2, y2], ...],
      "gt_rate": 0.224,
      "pred_rate": 0.476
    }
  ]
}
```

### Match Metrics
- **avg_distance**: Mean viewing angle offset (degrees) between matched points
- **distances**: Per-point angular distances
- **final_matches**: Paired points from hand-trace and ground truth
- **draw_traj**: Denoised hand-traced trajectory
- **gt_rate**: Ground truth coverage (matched GT points / total GT points)
- **pred_rate**: Tracing precision (matched traced points / total traced points)

## Installation

### Requirements
```bash
pip install numpy scipy matplotlib shapely opencv-python
pip install torch torchcpd
pip install tqdm pathlib
```

### Optional (for EXR analysis)
```bash
pip install OpenEXR Imath
```

## Usage

### 1. Process Individual Subject Data
```bash
python main.py
```
This will:
- Load hand-traced trajectories from `data/`
- Match them with ground truth annotations
- Generate visualizations in `results/`
- Save matched trajectory data

### 2. Batch Processing
```bash
python process.py
```
Processes all subjects with progress tracking.

### 3. Create Panoramas
```bash
python stitching.py
```
Generates panoramic views and homography matrices for each scene.

### 4. Statistical Analysis
Open `statistics.ipynb` in Jupyter to:
- Analyze detection and recognition rates
- Compare performance across conditions
- Generate publication-ready figures

## Key Algorithms

### Coherent Point Drift (CPD)
The core matching algorithm in `utils/registor.py`:

1. **Multi-phase resampling**: Tests different starting points
2. **Constrained rigid alignment**: Limits rotation and scaling
3. **Non-rigid refinement**: Deformable registration with regularization
4. **Hybrid matching**: 
   - Relaxed BBS for top-k candidates
   - Hungarian algorithm for remaining points
   - Distance threshold filtering

### Trajectory Processing
In `process.py` and `main.py`:

1. **Polygon extraction**: Converts nested point lists to valid polygons
2. **Boundary simplification**: Samples at regular intervals
3. **Coordinate transformation**: Flips y-axis, converts to viewing angles
4. **Validation**: Ensures polygon validity using Shapely

## Visualization Outputs

### Per-Subject Results (`results/SubXXX/`)
- **`nn_{ind}.jpg`**: CPD matching visualization with color-coded distances
- **`trajectory_{ind}.png`**: Overlay of hand-trace vs ground truth
- **Subdirectories**: Organized by visibility and lighting conditions

### Aggregate Visualizations (`figures/`)
- Detection and recognition rate plots
- Distance distribution histograms
- Coverage rate comparisons

## Citation

If you use this code or data, please cite:
```
[Your paper citation here]
```

## License

[Specify your license]

## Contact

For questions or issues, please contact:
- [Your name and email]

## Acknowledgments

This research was conducted at [Your institution] with support from [Funding sources].

## Notes

- All angles are in degrees
- Coordinate system: Origin at top-left, y-axis points down (image coordinates)
- Viewing angles are computed from camera intrinsics (fx=864.86, fy=11113.49)
- Distance threshold (tau) is adaptively set to median of top-k nearest distances
- GPU acceleration available for CPD (set device in `utils/registor.py`)
