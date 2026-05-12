# Extract Features — Low-Level Design

Detailed technical walkthrough of [extract_features.py](extract_features.py). Every function, constant, and design decision is explained so anyone can understand, modify, or extend the pipeline.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Constants & Configuration](#2-constants--configuration)
3. [DestrieuxAtlas Class](#3-destrieuxatlas-class)
4. [Feature Extraction Functions](#4-feature-extraction-functions)
5. [Orchestration Functions](#5-orchestration-functions)
6. [CLI & Entry Point](#6-cli--entry-point)
7. [How to Modify](#7-how-to-modify)

---

## 1. Architecture Overview

```
extract_features.py
│
├── CONSTANTS (lines 47-120)
│   ├── REGION_GROUPS          ← brain region → Destrieux label mappings
│   ├── KEY_INDIVIDUAL_REGIONS ← 6 most discriminative single regions
│   └── DIMENSION_WEIGHTS      ← weights for overall engagement score
│
├── DestrieuxAtlas (lines 127-172)
│   ├── __init__()             ← loads atlas once, pre-computes vertex indices
│   ├── vertices_for()         ← lookup indices for a single region
│   └── vertices_for_group()   ← lookup indices for a list of regions
│
├── FEATURE EXTRACTORS (lines 179-371)
│   ├── extract_dimension_features()    ← visual/auditory/emotional/language
│   ├── extract_attention_features()    ← hook ratio, onset speed
│   ├── extract_temporal_features()     ← slopes, peaks, sustained engagement
│   ├── extract_global_features()       ← total energy, mean, peak, duration
│   ├── extract_individual_region_features() ← fusiform, insula, broca, etc.
│   └── compute_weighted_scores()       ← overall engagement formula
│
├── PER-FILE PROCESSING (lines 378-439)
│   ├── extract_features_from_npz()     ← calls all 6 extractors for one file
│   └── extract_engagement_curve()      ← second-by-second curve (optional)
│
├── BATCH PROCESSING (lines 446-537)
│   ├── discover_npz_files()            ← recursive .npz file discovery
│   ├── run_batch()                     ← main loop: iterate files → build CSV
│   └── print_feature_summary()         ← log stats and NaN checks
│
└── CLI (lines 544-621)
    ├── parse_args()                    ← argparse setup
    └── main()                          ← entry point
```

---

## 2. Constants & Configuration

### `REGION_GROUPS` (dict, lines 47-101)

Maps each scoring dimension to a list of Destrieux atlas label strings. These labels are the exact names from the [Destrieux Surface Atlas](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_surf_destrieux.html) (76 labels, indices 0-75).

**Source of truth:** Region selections come from [Creative_Quality_Scorer_Design.md](../Creative_Quality_Scorer_Design.md) §3.1–3.5, which defines which brain regions map to each scoring dimension based on neuroscience literature.

**Structure:**
```python
REGION_GROUPS = {
    "visual": ["G_occipital_middle", "G_occipital_sup", ...],     # 9 regions
    "auditory": ["S_temporal_transverse", ...],                    # 6 regions
    "emotional": ["G_orbital", "S_orbital-H_Shaped", ...],        # 13 regions
    "language": ["G_front_inf-Opercular", ...],                    # 6 regions
}
```

**How it's used:** Each group is passed to `atlas.vertices_for_group()` to get all vertex indices for that group. Then `preds[:, verts]` slices the prediction matrix to get only those vertices.

### `KEY_INDIVIDUAL_REGIONS` (dict, lines 104-111)

Six individual atlas regions extracted separately because they are the most neuroscientifically discriminative:

```python
KEY_INDIVIDUAL_REGIONS = {
    "fusiform":     "G_oc-temp_lat-fusifor",    # face/object recognition
    "insula_short": "G_insular_short",          # emotional awareness
    "orbital":      "G_orbital",                # reward valuation
    "calcarine":    "S_calcarine",              # primary visual input
    "heschl":       "G_temp_sup-G_T_transv",    # core sound processing
    "broca":        "G_front_inf-Opercular",    # speech processing
}
```

### `DIMENSION_WEIGHTS` (dict, lines 114-120)

Weights for the overall engagement score formula:

```python
DIMENSION_WEIGHTS = {
    "visual": 0.25, "auditory": 0.15, "emotional": 0.30,
    "attention": 0.20, "language": 0.10,
}
```

**Rationale (from Creative_Quality_Scorer_Design.md §4):**
- Emotional is highest (30%) — emotion drives ad recall and action
- Visual is 25% — mobile ads are primarily visual
- Attention is 20% — if you don't hook in 3 seconds, nothing else matters
- Auditory is 15% — many users watch ads with sound off
- Language is 10% — game ads rely more on visuals than words

**To modify:** Change the weights and re-run. The overall score will be recalculated.

---

## 3. DestrieuxAtlas Class

**Purpose:** Load the Destrieux surface atlas once and provide O(1) vertex lookups by region name. This is an optimization — without it, every feature extraction call would need to re-load the atlas and re-compute vertex indices.

### `__init__(self)` (lines 133-157)

**What happens:**
1. Calls `nilearn.datasets.fetch_atlas_surf_destrieux()` — downloads the atlas on first run (~small), cached locally after that
2. Concatenates `map_left` and `map_right` hemisphere arrays into a single 20,484-element ROI map
3. Decodes label bytes to strings (nilearn returns them as `b'G_orbital'`)
4. Pre-computes a `_label_to_idx` dict mapping each label name → array of vertex indices

**Key data structures:**
```python
self.roi_map         # np.ndarray, shape (20484,) — each element is the region index for that vertex
self.labels          # list[str], length 76 — region names
self._label_to_idx   # dict[str, np.ndarray] — "G_orbital" → [3421, 3422, ..., 3598]
```

### `vertices_for(label: str)` (lines 159-164)

Returns the numpy array of vertex indices for a single region. Returns empty array if label not found (with a warning log).

### `vertices_for_group(labels: list[str])` (lines 166-172)

Concatenates vertex indices for multiple regions into one array. Used when extracting dimension-level features (e.g., all 9 visual regions combined).

---

## 4. Feature Extraction Functions

Each function takes the `preds` array (shape `(n_seconds, 20484)`) and returns a `dict[str, float]`.

### `extract_dimension_features(preds, atlas)` (lines 179-202)

**Purpose:** Compute mean, peak (max), and standard deviation for each of the 4 region groups.

**Logic step by step:**
1. For each dimension in `REGION_GROUPS`:
   - Get combined vertex indices via `atlas.vertices_for_group(region_labels)`
   - Slice: `preds[:, verts]` → shape `(n_seconds, n_verts_in_group)`
   - Compute mean across vertices per second: `np.mean(preds[:, verts], axis=1)` → shape `(n_seconds,)` — this is the time-series of that dimension's activation
   - From that time-series, compute:
     - `mean` = `np.mean(ts)` — average activation across all seconds
     - `peak` = `np.max(ts)` — highest activation at any second
     - `std` = `np.std(ts)` — variability of activation over time

**Output:** 12 features: `visual_mean`, `visual_peak`, `visual_std`, `auditory_mean`, ... etc.

**Visual diagram:**
```
preds (30, 20484)
   │
   │  slice columns for "visual" regions
   ▼
(30, ~2100)    ← ~2100 vertices belong to the 9 visual regions
   │
   │  mean across axis=1 (vertices)
   ▼
(30,)          ← one activation value per second = the visual time-series
   │
   ├── np.mean → visual_mean
   ├── np.max  → visual_peak
   └── np.std  → visual_std
```

---

### `extract_attention_features(preds)` (lines 205-243)

**Purpose:** Measure how quickly and powerfully the video grabs attention. This is NOT region-based — it uses **global** activation dynamics.

**Logic:**
1. Compute global time-series: `global_ts = np.mean(preds, axis=1)` — mean across ALL 20,484 vertices per second
2. Compute video-level stats: `video_mean`, `video_std`
3. **Hook ratio:** `mean(first 3 seconds) / mean(all seconds)`
   - If > 1.0 → the opening is stronger than the video average (good hook)
   - If < 1.0 → slow start, risk of viewer scrolling away
   - Edge case: if video_mean ≈ 0, defaults to 1.0
4. **Onset speed:** Find the first second where `global_ts > video_mean + 0.5 * video_std`
   - Lower value = the video grabs attention faster
   - If never exceeds threshold, returns `n_seconds` (no strong moment)

**Output:** 3 features: `attention_hook_ratio`, `attention_onset_second`, `attention_first3s_mean`

---

### `extract_temporal_features(preds)` (lines 246-305)

**Purpose:** Capture how engagement changes over time — are there peaks? Does it drop off? Is it sustained?

**Logic for each feature:**

1. **`peak_second`** = `np.argmax(global_ts)` — which second has the highest overall brain activation. Useful for finding the video's climax moment.

2. **`engagement_variance`** = `np.std(global_ts)` — high variance means the video has dynamic peaks and valleys; low variance means flat engagement.

3. **`engagement_slope_first_half`** and **`engagement_slope_second_half`**:
   - Split the time-series in half
   - Fit a linear regression (`np.polyfit(x, y, 1)`) to each half
   - The slope tells you if engagement is building (positive) or declining (negative)
   - Example: positive first-half slope + negative second-half slope = peaks in the middle

4. **`engagement_peak_count`**:
   - Uses `scipy.signal.find_peaks()` with `prominence = 0.5 * video_std`
   - Counts local maxima that are at least half a standard deviation above their neighbors
   - More peaks = more "wow moments" in the video

5. **`longest_sustained_above_mean`**:
   - Boolean array: `above_mean = global_ts > video_mean`
   - Walk through and find the longest consecutive `True` run
   - Longer runs = engagement is sustained, not just momentary spikes

6. **`pct_above_mean`** = fraction of seconds above the video's own mean activation

7. **`global_activation_range`** = `max(global_ts) - min(global_ts)` — total dynamic range

**Output:** 8 features

---

### `extract_global_features(preds)` (lines 308-318)

**Purpose:** Whole-brain, whole-video summary statistics.

**Logic:**
- `total_neural_energy` = `np.sum(np.maximum(preds, 0))` — sum of all positive activations. Negative values are zeroed out because positive BOLD signal indicates neural activity. This is a proxy for "how much total brain work happened."
- `global_mean_activation` = `np.mean(preds)` — average across everything
- `global_peak_activation` = `np.max(preds)` — single highest activation anywhere
- `video_duration_seconds` = `preds.shape[0]` — number of seconds processed
- `n_vertices` = `preds.shape[1]` — should always be 20,484 (sanity check)

**Output:** 5 features

---

### `extract_individual_region_features(preds, atlas)` (lines 321-339)

**Purpose:** Extract mean and peak for the 6 most neuroscientifically important individual regions (not grouped).

**Logic:** Same as dimension features but for a single atlas label instead of a group:
1. Get vertex indices for one label
2. Compute per-second mean: `np.mean(preds[:, verts], axis=1)`
3. Return mean and max of that time-series

**Why separate from dimension features?** These specific regions (fusiform, insula, etc.) often show up as top predictors in ML models. Having them as individual features lets XGBoost learn region-specific importance, rather than diluting their signal across a whole group.

**Output:** 12 features (6 regions × mean + peak)

---

### `compute_weighted_scores(features)` (lines 342-371)

**Purpose:** Compute the overall neural engagement score using the weighted formula from the Creative Quality Scorer design.

**Logic:**
1. For each dimension, grab the already-computed `_mean` feature as the raw score
2. Multiply each by its weight from `DIMENSION_WEIGHTS`
3. Sum for the overall score

**Important note about normalization:** These are RAW values, not 0-100 scores. True normalization requires:
1. Process all videos first
2. Find min/max for each dimension across the full dataset
3. Apply: `score = ((raw - min) / (max - min)) * 100`
4. Clamp to [0, 100]

This is deferred to a post-processing step (Phase 3 calibration).

**Output:** 6 features: `visual_score_raw`, `auditory_score_raw`, `emotional_score_raw`, `language_score_raw`, `attention_score_raw`, `overall_engagement_score_raw`

---

## 5. Orchestration Functions

### `extract_features_from_npz(npz_path, atlas)` (lines 378-413)

**Purpose:** Process one `.npz` file through all 6 extraction functions.

**Flow:**
```python
data = np.load(npz_path, allow_pickle=True)
preds = data["preds"]  # shape: (n_seconds, 20484)

features = {}
features.update(extract_dimension_features(preds, atlas))      # 12 features
features.update(extract_attention_features(preds))              # 3 features
features.update(extract_temporal_features(preds))               # 8 features
features.update(extract_global_features(preds))                 # 5 features
features.update(extract_individual_region_features(preds, atlas)) # 12 features
features.update(compute_weighted_scores(features))              # 6 features
                                                          # Total: 46 features
```

### `extract_engagement_curve(npz_path, atlas)` (lines 416-439)

**Purpose:** Produce a second-by-second DataFrame for visualization (not for ML).

**Output columns:** `second`, `global`, `visual`, `auditory`, `emotional`, `language`

This is only called when `--save-curves` is passed. Each video gets a separate CSV file.

### `run_batch(npz_files, atlas, output_csv, ...)` (lines 453-504)

**Purpose:** Main loop — iterates all `.npz` files, extracts features, builds a DataFrame, writes CSV.

**Flow:**
1. Loop through each `.npz` file
2. Call `extract_features_from_npz()` — returns a dict
3. Add metadata: `filename` (stem) and `video_folder` (parent dir name)
4. Append to `rows` list
5. Optionally save engagement curve
6. Build DataFrame, reorder columns (metadata first), write CSV

**Error handling:** If one file fails, it logs the error and continues to the next file. The batch doesn't stop for individual failures.

---

## 6. CLI & Entry Point

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--npz-dir` | One of these | — | Directory with .npz files |
| `--video-dir` | is required | — | Directory with videos + .npz files |
| `--output`, `-o` | No | `brain_features.csv` | Output CSV path |
| `--save-curves` | No | False | Save per-video engagement curves |
| `--curves-dir` | No | Same as .npz | Where to save curve CSVs |

### `main()` (lines 581-620)

1. Parse CLI arguments
2. Discover `.npz` files recursively under the root directory
3. Load the Destrieux atlas (once — reused for all files)
4. Call `run_batch()` to process all files
5. Print feature summary with stats and NaN check

---

## 7. How to Modify

### Add a new brain region group

1. Add the group to `REGION_GROUPS` with exact Destrieux label names
2. It will automatically get `_mean`, `_peak`, `_std` features in `extract_dimension_features()`
3. If you want it in the weighted score, add it to `DIMENSION_WEIGHTS` and `dim_keys` in `compute_weighted_scores()`

### Add a new individual region

1. Add to `KEY_INDIVIDUAL_REGIONS` with `short_name: "atlas_label"`
2. It will automatically get `_mean` and `_peak` features

### Change the scoring weights

Edit `DIMENSION_WEIGHTS`. The overall score will be recalculated on next run.

### Add a new temporal feature

Add to `extract_temporal_features()`. Use the `global_ts` time-series (already computed) to derive new features. Return them in the dict.

### Change the hook window from 3 seconds

In `extract_attention_features()`, change the slice `global_ts[:min(3, n_seconds)]` to your desired window.

### Add 0-100 normalization

Currently deferred to Phase 3. To add it here:
1. After `run_batch()` produces the DataFrame, compute `min`/`max` per raw score column
2. Apply: `df[col + "_normalized"] = ((df[col] - col_min) / (col_max - col_min)) * 100`
3. Clamp with `df[col].clip(0, 100)`

### Verify atlas labels

Run this to list all available Destrieux labels:
```python
from nilearn import datasets
atlas = datasets.fetch_atlas_surf_destrieux()
for i, l in enumerate(atlas['labels']):
    name = l.decode('utf-8') if isinstance(l, bytes) else l
    print(f"{i}: {name}")
```

### Run for a single file

```python
from extract_features import DestrieuxAtlas, extract_features_from_npz
from pathlib import Path

atlas = DestrieuxAtlas()
features = extract_features_from_npz(Path("video.npz"), atlas)
print(features)
```
