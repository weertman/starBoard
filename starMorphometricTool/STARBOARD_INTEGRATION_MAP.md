# starBoard Integration Map

This document maps the critical features of starMorphometricTool and how they integrate with starBoard.

## Overview

The starMorphometricTool provides webcam-based morphometric measurements for sunflower sea stars. When integrated with starBoard, it enables:

- **Live data capture**: Enter metadata while taking images
- **Quick comparison**: Compare captured stars with the database immediately  
- **Automated measurements**: YOLO-detected arm counts and calibrated measurements
- **Volume estimation**: Optional 3D volume calculation using Depth-Anything-V2
- **Dual storage**: Preserves full morphometric data while adding images to starBoard archive

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         starBoard Application                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌─────────────────┐  ┌────────────┐  ┌─────────────┐ │
│  │ TabSetup │  │ TabMorphometric │  │ TabFirst   │  │ TabSecond   │ │
│  └──────────┘  │    (NEW)        │  │ Order      │  │ Order       │ │
│                └────────┬────────┘  └─────▲──────┘  └─────────────┘ │
│                         │                 │                          │
│                         │ _notify_first_order_refresh()              │
│                         │                 │                          │
├─────────────────────────┼─────────────────┼──────────────────────────┤
│                         ▼                 │                          │
│               ┌─────────────────┐         │                          │
│               │ src/morphometric│         │                          │
│               │  __init__.py    │         │                          │
│               │  camera_adapter │         │                          │
│               │  detection_adpt │         │                          │
│               │  analysis_adpt  │         │                          │
│               │  depth_adapter  │         │                          │
│               │  data_bridge    │         │                          │
│               └────────┬────────┘         │                          │
│                        │                  │                          │
└────────────────────────┼──────────────────┼──────────────────────────┘
                         │                  │
           ┌─────────────▼──────────────┐   │
           │   starMorphometricTool     │   │
           │                            │   │
           │  camera/      Camera HAL   │   │
           │  detection/   YOLO + CB    │   │
           │  morphometrics/ Analysis   │   │
           │  depth/       DA-V2 Volume │   │
           │  ui/          PolarCanvas  │   │
           └─────────────┬──────────────┘   │
                         │                  │
           ┌─────────────▼──────────────────▼───────────────────┐
           │                  Data Storage                      │
           ├────────────────────────────────────────────────────┤
           │                                                    │
           │  measurements/                archive/             │
           │    └─gallery/                   └─gallery/         │
           │        └─{id}/                      └─{id}/        │
           │            └─{date}/                    └─{enc}/   │
           │                └─mFolder_N/               └─*.png  │
           │                    ├─raw_frame.png                 │
           │                    ├─corrected_*.png    metadata/  │
           │                    ├─morphometrics.json   └─*.csv  │
           │                    ├─corrected_detection.json      │
           │                    ├─calibrated_depth.npy (opt)    │
           │                    └─elevation_visualization.png   │
           │                                                    │
           └────────────────────────────────────────────────────┘
```

---

## Component Mapping

### Camera Module

| MorphometricTool Component | starBoard Integration | Purpose |
|---------------------------|----------------------|---------|
| `camera/interface.py` | `CameraAdapter` | Abstract camera interface |
| `camera/factory.py` | `CameraAdapter.initialize()` | Camera creation and auto-detection |
| `camera/providers/opencv_camera.py` | Via factory | OpenCV webcam implementation |
| `camera/config.py` | Config save/load | Persistent camera settings |

**Integration Points:**
- `CameraAdapter` wraps the camera module for starBoard use
- Lazy loading - only initialized when Morphometric tab is accessed
- Auto-detection finds first available camera

### Detection Module

| MorphometricTool Component | starBoard Integration | Purpose |
|---------------------------|----------------------|---------|
| `detection/yolo_handler.py` | `DetectionAdapter.predict()` | YOLO model loading and inference |
| `detection/yolo_handler.select_primary_detection()` | `DetectionAdapter.get_primary_detection()` | Extract best detection |
| `detection/checkerboard.py` | `DetectionAdapter.detect_checkerboard()` | Calibration board detection |

**Integration Points:**
- YOLO model loaded from `starMorphometricTool/models/best.pt`
- Checkerboard calibration provides `mm_per_pixel` scale
- Homography matrix enables perspective correction

### Morphometrics Module

| MorphometricTool Component | starBoard Integration | Purpose |
|---------------------------|----------------------|---------|
| `morphometrics/analysis.py` | `AnalysisAdapter.analyze_contour()` | Arm detection, area calculation |
| `morphometrics/analysis.find_arm_tips()` | Peak detection logic | Find arm tip coordinates |

**Integration Points:**
- Interactive parameter adjustment (smoothing, prominence, distance)
- Arm rotation for consistent numbering
- Real-time measurement updates

### UI Components

| MorphometricTool Component | starBoard Integration | Purpose |
|---------------------------|----------------------|---------|
| `ui/components/polar_canvas.py` | Direct import in TabMorphometric | Interactive arm tip editing |

**Integration Points:**
- Click to add peaks, Shift+click to remove
- Signal `peaksChanged` triggers measurement recalculation
- Visual feedback of arm detection results

### Depth Module (Optional)

| MorphometricTool Component | starBoard Integration | Purpose |
|---------------------------|----------------------|---------|
| `depth/depth_handler.py` | `DepthAdapter.run_volume_estimation()` | Depth-Anything-V2 model loading and inference |
| `depth/volume_estimation.py` | Via DepthAdapter | Depth calibration and volume computation |

**Integration Points:**
- Uses Depth-Anything-V2 (vitb encoder by default) for monocular depth estimation
- Calibrates depth using checkerboard corners as reference plane
- Computes volume from elevation above reference plane
- Saves depth data (calibrated_depth.npy, elevation_visualization.png) to mFolder
- Graceful fallback when Depth-Anything-V2 is not installed

**Requirements:**
- `Depth-Anything-V2/` directory alongside starMorphometricTool
- Model checkpoint at `Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth`
- PyTorch (CUDA optional, falls back to CPU)

---

## Data Flow

### On Save

```
1. User clicks "Save to starBoard"
   │
   ├─2. AnalysisAdapter.save_measurement()
   │    │
   │    └─► measurements/{type}/{id}/{date}/mFolder_N/
   │           ├─ raw_frame.png
   │           ├─ corrected_mask.png
   │           ├─ corrected_object.png
   │           ├─ checkerboard_with_object.png
   │           ├─ morphometrics.json
   │           └─ corrected_detection.json
   │
   ├─3. place_images() copies raw_frame.png
   │    │
   │    └─► archive/{gallery|queries}/{id}/{encounter}/raw_frame.png
   │
   ├─4. append_row() adds metadata
   │    │
   │    └─► metadata/{gallery|queries}_metadata.csv
   │           └─ Row with morph_* fields + standard fields
   │
   └─5. _notify_first_order_refresh()
        │
        └─► TabFirstOrder rebuilds and refreshes UI
```

---

## Field Mappings

### Morphometric → starBoard Schema

| morphometrics.json Field | starBoard Field | Type | Description |
|-------------------------|-----------------|------|-------------|
| `num_arms` | `morph_num_arms` | INT | YOLO-detected arm count |
| `area_mm2` | `morph_area_mm2` | FLOAT | Calibrated surface area |
| `major_axis_mm` | `morph_major_axis_mm` | FLOAT | Fitted ellipse major axis |
| `minor_axis_mm` | `morph_minor_axis_mm` | FLOAT | Fitted ellipse minor axis |
| `arm_data[*][3]` | `morph_mean_arm_length_mm` | FLOAT | Average arm length |
| `max(arm_data[*][3])` | `morph_max_arm_length_mm` | FLOAT | Longest arm |
| *calculated* | `morph_tip_to_tip_mm` | FLOAT | Max opposing tip distance |
| `volume_estimation.volume_mm3` | `morph_volume_mm3` | FLOAT | Optional volume estimate |
| *mFolder path* | `morph_source_folder` | TEXT | Traceability reference |

### arm_data Structure

Each entry in `arm_data` is: `[arm_number, x_vec, y_vec, length_mm]`

- `arm_number`: 1-indexed arm ID (after rotation)
- `x_vec`, `y_vec`: Vector from center to tip (mm)
- `length_mm`: Arm length in millimeters

---

## Safety Measures

### Isolation Strategy

1. **Lazy Loading**: Morphometric adapters only initialized when tab accessed
2. **Conditional Import**: Tab gracefully falls back if dependencies unavailable
3. **Try/Except Wrapping**: All morphometric operations isolated from main app
4. **Dual-Save Order**: mFolder saved first (atomic), then archive copy

### Feature Flag

The integration can be disabled by:
- Removing or renaming `starMorphometricTool/` directory
- Uninstalling `ultralytics` package
- Catching ImportError in `src/morphometric/__init__.py`

### Data Integrity

- **Append-only**: No modification of existing metadata rows
- **Original preserved**: raw_frame.png copied, not moved
- **Full backup**: mFolder contains complete analysis data

---

## Dependencies

### Required for Morphometric Tab

```
opencv-python>=4.5.0
ultralytics>=8.0.0
scipy>=1.7.0
matplotlib>=3.5.0
numpy>=1.21.0
```

### Optional for Volume Estimation

```
torch>=2.0.0           # PyTorch (CUDA optional)
Depth-Anything-V2/     # Clone from GitHub
  └─ checkpoints/
      └─ depth_anything_v2_vitb.pth  # Download from HuggingFace
```

### Graceful Degradation

If dependencies are missing:
- Morphometric tab shows installation instructions
- Volume estimation button shows error if Depth-Anything-V2 unavailable
- Other starBoard functionality unaffected
- No error on application startup

---

## File Structure Summary

### New Files in starBoard

```
src/
  └─ morphometric/
      ├─ __init__.py          # Module init, lazy loading
      ├─ camera_adapter.py    # Camera subsystem wrapper
      ├─ detection_adapter.py # YOLO/checkerboard wrapper
      ├─ analysis_adapter.py  # Morphometrics wrapper
      ├─ depth_adapter.py     # Depth-Anything-V2 wrapper (optional)
      └─ data_bridge.py       # JSON → starBoard field mapping

  └─ ui/
      └─ tab_morphometric.py  # Main tab UI (new)

  └─ data/
      └─ annotation_schema.py # +9 morph_* fields, +1 field group
```

### Modified Files

```
src/ui/main_window.py  # Conditional tab insertion
```

---

## Usage Workflow

1. **Start Stream**: Click "Start Stream" to view webcam feed
2. **Calibrate**: Position checkerboard, click "Detect Board"
3. **Position Star**: Place star in view, ensure good detection
4. **Enable Detection**: Click "Start Detection" for live YOLO
5. **Capture**: Click "Capture Detection" when satisfied
6. **Analyze**: Click "Run Analysis" to measure morphometrics
7. **Adjust**: Use sliders to refine arm detection
8. **Edit Tips**: Click polar plot to add/remove arm tips
9. **Volume** (optional): Click "Estimate Volume" for 3D volume calculation
10. **Enter Metadata**: Fill in ID, location, initials
11. **Save**: Click "Save to starBoard" for dual-save

---

## Future Considerations

- **Batch Processing**: Load existing mFolders into starBoard
- **Auto-population**: Use morph_num_arms to suggest num_total_arms
- **Model Updates**: Support for custom YOLO models per deployment
- **Depth Camera**: Direct depth camera integration (RealSense, Kinect) for improved accuracy

