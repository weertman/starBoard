# Star Dataset Utilities

This folder contains utility scripts for managing and processing the Star Identification dataset.

## Scripts

### `process_raw_data.py`

**Purpose:**  
Automates the conversion of raw field images into the standardized format required for the ReID training pipeline. It detects stars using the YOLO segmentation model, removes the background, crops the image tightly, and saves it to the processed dataset folder.

**Features:**
- **YOLOv8 Segmentation:** Automatically finds and isolates the star.
- **High Performance:** Optimized for speed with:
    - **Batch Processing:** Processes multiple images simultaneously on the GPU.
    - **Crop-First Strategy:** Minimizes memory usage by cropping before masking.
    - **Parallel I/O:** Uses multi-threaded loading and saving to prevent disk bottlenecks.
    - **High RAM Mode:** Aggressively pre-loads data into memory.

**Usage:**

Run the script from the project root:

```bash
python star_dataset_utils/process_raw_data.py DATASET_NAME [options]
```

**Arguments:**

- `DATASET_NAME`: The name of the folder inside `star_dataset_raw` (e.g., `FHL_EAGLE_PT_PYCNOS`).
- `--batch-size`: Number of images to process at once (Default: 32). Increase this if you have a powerful GPU/CPU.
- `--confidence`: Minimum confidence threshold for detection (Default: 0.7).
- `--keep-failed`: If set, copies the original image if segmentation fails (instead of skipping it).

**Example:**

```bash
# Process a new dataset with a batch size of 64
python star_dataset_utils/process_raw_data.py FHL_EAGLE_PT_PYCNOS --batch-size 64
```

## Workflow

1.  **Add Data:** Place your new image folders into `star_dataset_raw/DATASET_NAME/...` following the project structure (Individual/Date/Image).
2.  **Process:** Run `process_raw_data.py` to generate the clean images in `star_dataset/`.
3.  **Update Metadata:** After processing, run the metadata generation script to update the training splits:
    ```bash
    python -m temporal_reid.data.prepare --dataset-root ./star_dataset --force
    ```


