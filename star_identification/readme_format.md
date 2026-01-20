# Star Dataset Format Specification

This document describes the required directory structure and naming conventions for adding new data to the `star_dataset_raw` folder.

---

## Directory Structure

The dataset follows a strict **three-level hierarchy**:

```
star_dataset_raw/
└── DATASET_NAME/
    └── individual_id/
        └── date_folder/
            └── image_files
```

### Level 1: Dataset Name (`DATASET_NAME`)

The top-level folder represents a **collection source** or **study context**.

**Examples:**
- `ADULT_FHL_STARS` – Named adult stars at Friday Harbor Labs
- `HodinLab_2023_BroodInResidence` – Hodin Lab brood study 2023
- `PWS_2023` – Prince William Sound survey 2023
- `WILD_STARS_FHL_DOCK` – Wild star observations from FHL dock

**Naming conventions:**
- Use UPPERCASE or CamelCase for readability
- Include year if the dataset is time-bounded
- Use underscores `_` to separate words
- Avoid spaces and special characters

---

### Level 2: Individual ID (`individual_id`)

Each subfolder represents a **unique individual** star.

**Examples:**
- `stella` – A named individual star
- `ursa_minor` – Another named individual
- `r2z6` – Lab-assigned identifier
- `unknown_001` – Unidentified individual (use sequential numbering)

**Naming conventions:**
- Use lowercase with underscores
- Keep names short but descriptive
- For unnamed individuals, use consistent identifiers (e.g., `ind_001`, `star_42`)
- The combination `DATASET_NAME__individual_id` becomes the unique identity in the system

---

### Level 3: Date Folder (`date_folder`) — CRITICAL FOR TEMPORAL SPLITTING

Each subfolder represents a **single observation session (outing)**.

**Format:** `M_D_YYYY_description`

| Component | Format | Example |
|-----------|--------|---------|
| Month | 1-12 (no leading zero) | `3` for March |
| Day | 1-31 (no leading zero) | `23` |
| Year | 4 digits | `2024` |
| Description | lowercase with underscores | `dock_sighting` |

**Examples:**
```
3_23_2024_dock_sighting_ww      ✓ Good
3_23_2024_dock_sighting_ju      ✓ Good (different photographer same day)
6_10_2022_star_card_and_stars   ✓ Good
11_02_2024_48ftMLLWBelowYogaDeck ✓ Good
4_6_2024_brown_island_release   ✓ Good

2024-03-23_dock                 ✗ Wrong format (use underscores, not dashes)
March_23_2024_dock              ✗ Wrong format (use numeric month)
dock_sighting_march             ✗ Missing date
```

**Why this matters:**
The temporal re-identification system parses dates from folder names to:
1. Sort outings chronologically
2. Split data into train (earlier) and test (later) sets
3. Ensure temporal separation between training and evaluation

**If date is unknown:**
Use a descriptive name without the date prefix. These images will be assigned to training only and marked as having no temporal information:
```
unknown_date_lab_photos/
field_collection_batch_1/
```

---

### Level 4: Image Files

The actual image files of the star.

**Supported formats:**
- `.png` (preferred)
- `.jpg` / `.jpeg`
- `.JPG` / `.JPEG`

**Naming conventions:**
- Any filename is acceptable
- Original camera filenames are fine (e.g., `DSC01105.png`, `IMG_3132.png`)
- Frame extracts from video are fine (e.g., `GX010026 - frame at 0m19s.png`)
- Descriptive names are helpful but not required

---

## Complete Example

```
star_dataset_raw/
├── MY_NEW_SURVEY_2024/
│   ├── star_alpha/
│   │   ├── 5_15_2024_initial_observation/
│   │   │   ├── IMG_0001.png
│   │   │   ├── IMG_0002.png
│   │   │   └── IMG_0003.png
│   │   └── 6_20_2024_followup_dive/
│   │       ├── DSC_0100.png
│   │       └── DSC_0101.png
│   ├── star_beta/
│   │   └── 5_15_2024_initial_observation/
│   │       └── IMG_0010.png
│   └── unknown_001/
│       └── 5_16_2024_transect_survey/
│           ├── frame_001.png
│           └── frame_002.png
```

This would create identities:
- `MY_NEW_SURVEY_2024__star_alpha` (2 outings, 5 images) → **evaluable**
- `MY_NEW_SURVEY_2024__star_beta` (1 outing, 1 image) → **negative-only**
- `MY_NEW_SURVEY_2024__unknown_001` (1 outing, 2 images) → **negative-only**

---

## Processing Pipeline

After adding new data to `star_dataset_raw`:

1. **Preprocessing** (if needed): Convert images to PNG, crop to star region
2. **Copy to `star_dataset/`**: The processed dataset folder
3. **Regenerate metadata**:
   ```bash
   conda activate wildlife
   cd D:\star_identification
   python -m temporal_reid.data.prepare --dataset-root ./star_dataset --force
   ```

---

## Important Notes

### For Temporal Evaluation

An individual needs **at least 2 outings** (different `date_folder` entries) to be used for temporal evaluation. Individuals with only 1 outing are used as **negative examples only** during training.

### Multiple Photographers Same Day

If multiple photographers capture the same individual on the same day, create separate date folders with a suffix:
```
3_23_2024_dock_sighting_ww/   # Photographer WW
3_23_2024_dock_sighting_ju/   # Photographer JU
```

These are treated as **separate outings** even though they share the same calendar date.

### Re-sightings Are Valuable

The most valuable data for re-identification training is **the same individual observed on multiple different days**. Prioritize:
1. Adding new observation dates for existing individuals
2. Including diverse conditions (lighting, angle, water clarity)
3. Capturing images across seasons if possible

### Image Quality Guidelines

- Clear view of the aboral (top) surface
- Star should fill a reasonable portion of the frame
- Multiple angles per session are helpful
- Both close-up detail and full-body shots are useful

---

## Summary Checklist

Before adding new data, verify:

- [ ] Dataset folder has a descriptive UPPERCASE name
- [ ] Individual folders use lowercase with underscores
- [ ] Date folders follow `M_D_YYYY_description` format
- [ ] Images are in PNG, JPG, or JPEG format
- [ ] Each date folder represents a single observation session
- [ ] Re-sighted individuals have multiple date folders (when possible)



