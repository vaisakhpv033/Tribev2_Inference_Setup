# Phase 2 Pipeline: Brain Feature Extraction

## Overview

Phase 2 handles end-to-end extraction of brain features from ad videos. It takes raw `.mp4` files, runs them through a GPU-based TRIBEv2 model, and produces a structured CSV of brain activity features ready for machine learning.

```
 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
 │   Ad Videos     │       │   TRIBEv2 API   │       │  Brain Feature  │
 │   (.mp4 files)  │──────▶│   (RunPod GPU)  │──────▶│  CSV Output     │
 │                 │ HTTP  │  Celery + Redis  │ .npz  │  1 row/video    │
 │  720x1280/      │       │  + FastAPI       │       │  46 features    │
 │  720x900/       │       │                 │       │                 │
 └─────────────────┘       └─────────────────┘       └─────────────────┘
    batch_analyze.py          (RunPod pod)          extract_features.py
```

---

## Pipeline Components

### Step 1: Batch Video Upload — `batch_analyze.py`

Discovers all `.mp4` files, uploads each to the TRIBEv2 API on RunPod, polls until inference completes, and downloads the `.npz` result. The `.npz` is saved alongside the original video with the same name.

**Key features:**
- **Resume-safe** — progress saved to `batch_state.json`; re-running picks up where it left off
- **Skip existing** — won't re-process videos that already have a `.npz` file
- **Sequential** — one video at a time (GPU is the bottleneck)

```bash
python batch_analyze.py --base-url https://<pod-id>-8000.proxy.runpod.net
```

**Output per video:** `.npz` file containing `preds` array of shape `(n_seconds, 20484)`.

---

### Step 2: TRIBEv2 Inference API (RunPod)

FastAPI + Celery on a GPU pod. See [README.md](README.md) for full API docs.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/jobs/analyze` | POST | Upload video → get job_id |
| `/api/v1/jobs/{job_id}/status` | GET | Poll job status |
| `/api/v1/jobs/{job_id}/result` | GET | Download .npz result |

---

### Step 3: Feature Extraction — `extract_features.py`

Processes all `.npz` files, maps 20,484 vertices to named brain regions via the **Destrieux Atlas**, outputs a single CSV with **1 row per video, 46 features**.

```bash
python extract_features.py \
    --video-dir "path/to/Marketing Final Videos" \
    --output brain_features.csv
```

**Why one row per video?** The downstream model is XGBoost (tabular), not LSTM. Campaign metadata is per-video. Temporal dynamics are captured *as summary features*.

---

## Feature Categories (46 Total)

### 1. Per-Dimension Stats (12 features)

| Dimension | Regions | Features |
|---|---|---|
| **Visual** | 9 occipital + fusiform | `visual_mean`, `visual_peak`, `visual_std` |
| **Auditory** | 6 superior temporal | `auditory_mean`, `auditory_peak`, `auditory_std` |
| **Emotional/Reward** | 13 orbitofrontal + insula + cingulate | `emotional_mean`, `emotional_peak`, `emotional_std` |
| **Language** | 6 Broca's + Wernicke's | `language_mean`, `language_peak`, `language_std` |

### 2. Attention / Hook (3 features)

| Feature | Meaning |
|---|---|
| `attention_hook_ratio` | First 3s activation vs video average (>1 = strong hook) |
| `attention_onset_second` | First second above threshold (lower = faster grab) |
| `attention_first3s_mean` | Raw mean activation in first 3 seconds |

### 3. Temporal Dynamics (8 features)

| Feature | Meaning |
|---|---|
| `peak_second` | Second with highest global activation |
| `engagement_variance` | How dynamic the brain response is over time |
| `engagement_slope_first_half` | Engagement trend in first half |
| `engagement_slope_second_half` | Engagement trend in second half |
| `engagement_peak_count` | Number of "wow moments" (local maxima) |
| `longest_sustained_above_mean` | Longest consecutive run above average |
| `pct_above_mean` | % of video above average activation |
| `global_activation_range` | Max - min global activation |

### 4. Global Stats (5 features)

| Feature | Meaning |
|---|---|
| `total_neural_energy` | Sum of all positive activations |
| `global_mean_activation` | Average across all vertices and seconds |
| `global_peak_activation` | Single highest activation value |
| `video_duration_seconds` | Video duration in seconds |
| `n_vertices` | Always 20,484 (sanity check) |

### 5. Key Individual Regions (12 features)

Mean + peak for the 6 most discriminative regions:

| Prefix | Region | Role |
|---|---|---|
| `fusiform_` | G_oc-temp_lat-fusifor | Face/object recognition |
| `insula_short_` | G_insular_short | Emotional awareness |
| `orbital_` | G_orbital | Reward valuation |
| `calcarine_` | S_calcarine | Primary visual input |
| `heschl_` | G_temp_sup-G_T_transv | Core sound processing |
| `broca_` | G_front_inf-Opercular | Speech processing |

### 6. Weighted Scores (6 features)

```
Overall = (Visual × 0.25) + (Auditory × 0.15) + (Emotional × 0.30)
        + (Attention × 0.20) + (Language × 0.10)
```

`visual_score_raw`, `auditory_score_raw`, `emotional_score_raw`, `language_score_raw`, `attention_score_raw`, `overall_engagement_score_raw`

> **Note:** Raw values, not normalized to 0-100. True normalization needs a calibration baseline (Phase 3).

---

## Output Files

| File | Description |
|---|---|
| `brain_features.csv` | Main output — 1 row/video, 48 columns (2 metadata + 46 features) |
| `batch_state.json` | Resume state for batch_analyze.py |
| `curves/*.csv` | Optional per-video engagement curves for visualization |

---

## What Comes Next (Phase 3)

1. **Data merging** — join brain features with campaign metadata (spend, region, installs)
2. **Calibration** — normalize raw scores to 0-100 scale
3. **Model training** — XGBoost regression on merged feature matrix
4. **Feature importance** — quantify brain features' contribution vs campaign metadata
