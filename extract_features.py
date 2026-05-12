"""
TRIBEv2 Brain Feature Extraction Pipeline
==========================================

Processes .npz files produced by TRIBEv2 inference and extracts ~51 brain
features per video using the Destrieux Surface Atlas.  Outputs a single CSV
with one row per video — ready for XGBoost / LightGBM training.

Usage:
    python extract_features.py --npz-dir <DIR_WITH_NPZ_FILES>
    python extract_features.py --npz-dir <DIR> --output brain_features.csv
    python extract_features.py --npz-dir <DIR> --save-curves

The script can also auto-discover .npz files next to video files:
    python extract_features.py --video-dir <DIR_WITH_VIDEOS_AND_NPZ>
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("extract_features")


# ═══════════════════════════════════════════════════════════════════════════════
# BRAIN REGION DEFINITIONS  (from Creative_Quality_Scorer_Design.md)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each dimension maps to specific Destrieux atlas label names.
# Labels were cross-checked against the actual atlas (76 labels, indices 0-75).
#

REGION_GROUPS = {
    # ── Visual Engagement ────────────────────────────────────────────────
    # Occipital lobe + fusiform: core visual processing, motion, faces
    "visual": [
        "G_occipital_middle",           # Middle Occipital Gyrus
        "G_occipital_sup",              # Superior Occipital Gyrus
        "Pole_occipital",               # Occipital Pole (V1)
        "G_and_S_occipital_inf",        # Inferior Occipital
        "G_cuneus",                     # Cuneus
        "S_calcarine",                  # Calcarine Sulcus (primary visual input)
        "S_oc_middle_and_Lunatus",      # Lunate Sulcus
        "S_oc_sup_and_transversal",     # Superior Occipital Sulcus
        "G_oc-temp_lat-fusifor",        # Fusiform Gyrus (faces/objects)
    ],

    # ── Auditory Engagement ──────────────────────────────────────────────
    # Superior temporal regions: sound, music, voice, speech
    "auditory": [
        "S_temporal_transverse",        # Transverse Temporal Sulcus
        "G_temp_sup-G_T_transv",        # Heschl's Gyrus (core sound)
        "G_temp_sup-Lateral",           # Superior Temporal Gyrus
        "G_temp_sup-Plan_tempo",        # Planum Temporale (music/speech)
        "S_temporal_sup",               # Superior Temporal Sulcus
        "G_temp_sup-Plan_polar",        # Planum Polare (melody/pitch)
    ],

    # ── Emotional / Reward (Dopamine Proxy) ──────────────────────────────
    # Orbitofrontal + Insula + Anterior Cingulate: emotion, reward, desire
    "emotional": [
        "G_orbital",                    # Orbital Gyrus (reward valuation)
        "S_orbital-H_Shaped",           # H-Shaped Orbital Sulcus (decisions)
        "S_orbital_med-olfact",         # Medial Orbital Sulcus (emotion)
        "S_orbital_lateral",            # Lateral Orbital Sulcus (reward expectation)
        "G_rectus",                     # Gyrus Rectus (emotional regulation)
        "S_circular_insula_inf",        # Inferior Circular Insular (gut feelings)
        "S_circular_insula_sup",        # Superior Circular Insular (subjective experience)
        "S_circular_insula_ant",        # Anterior Circular Insular (anticipation)
        "G_insular_short",              # Short Insular Gyrus (emotional awareness)
        "G_Ins_lg_and_S_cent_ins",      # Long Insular Gyrus (pain/pleasure)
        "G_and_S_cingul-Ant",           # Anterior Cingulate (emotional conflict)
        "G_subcallosal",                # Subcallosal Gyrus (mood/reward)
        "G_front_inf-Orbital",          # Inferior Frontal Orbital (impulse/desire)
    ],

    # ── Language / Narrative ─────────────────────────────────────────────
    # Broca's + Wernicke's + temporal language: speech, reading, narrative
    "language": [
        "G_front_inf-Opercular",        # Broca's Area (Opercular)
        "G_front_inf-Triangul",         # Broca's Area (Triangular)
        "G_temporal_middle",            # Middle Temporal Gyrus (word meaning)
        "G_temporal_inf",               # Inferior Temporal Gyrus (reading)
        "S_front_inf",                  # Inferior Frontal Sulcus (verbal WM)
        "G_and_S_cingul-Mid-Ant",       # Mid-Anterior Cingulate (verbal attention)
    ],
}

# Key individual regions to extract separately (most discriminative)
KEY_INDIVIDUAL_REGIONS = {
    "fusiform":  "G_oc-temp_lat-fusifor",     # Face/object recognition
    "insula_short": "G_insular_short",         # Emotional awareness
    "orbital":   "G_orbital",                  # Reward valuation
    "calcarine": "S_calcarine",                # Primary visual input
    "heschl":    "G_temp_sup-G_T_transv",      # Core sound processing
    "broca":     "G_front_inf-Opercular",      # Speech processing
}

# Overall score weights (from Creative_Quality_Scorer_Design.md §4)
DIMENSION_WEIGHTS = {
    "visual":    0.25,
    "auditory":  0.15,
    "emotional": 0.30,
    "attention": 0.20,
    "language":  0.10,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ATLAS LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class DestrieuxAtlas:
    """
    Loads the Destrieux surface atlas once and provides fast vertex-index
    lookups for named regions.
    """

    def __init__(self):
        from nilearn import datasets
        log.info("Loading Destrieux surface atlas…")
        atlas = datasets.fetch_atlas_surf_destrieux()

        # Combined left + right hemisphere ROI map  (20,484 vertices)
        self.roi_map = np.concatenate([atlas["map_left"], atlas["map_right"]])

        # Decode label bytes → str
        self.labels = [
            l.decode("utf-8") if isinstance(l, bytes) else l
            for l in atlas["labels"]
        ]

        # Pre-compute vertex indices for every label
        self._label_to_idx: dict[str, np.ndarray] = {}
        for roi_idx, name in enumerate(self.labels):
            verts = np.where(self.roi_map == roi_idx)[0]
            if len(verts) > 0:
                self._label_to_idx[name] = verts

        log.info(
            "Atlas ready: %d labels, %d vertices",
            len(self.labels), len(self.roi_map),
        )

    def vertices_for(self, label: str) -> np.ndarray:
        """Return vertex indices for a single region label."""
        if label not in self._label_to_idx:
            log.warning("Label '%s' not found in atlas — skipping", label)
            return np.array([], dtype=int)
        return self._label_to_idx[label]

    def vertices_for_group(self, labels: list[str]) -> np.ndarray:
        """Return combined vertex indices for a list of region labels."""
        parts = [self.vertices_for(l) for l in labels]
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dimension_features(
    preds: np.ndarray,
    atlas: DestrieuxAtlas,
) -> dict[str, float]:
    """
    Per-dimension features: mean, peak, std across all seconds for each
    region group (visual, auditory, emotional, language).
    """
    features = {}
    for dim_name, region_labels in REGION_GROUPS.items():
        verts = atlas.vertices_for_group(region_labels)
        if len(verts) == 0:
            for stat in ("mean", "peak", "std"):
                features[f"{dim_name}_{stat}"] = np.nan
            continue

        # shape: (n_seconds,) — mean activation across group vertices per second
        ts = np.mean(preds[:, verts], axis=1)

        features[f"{dim_name}_mean"] = float(np.mean(ts))
        features[f"{dim_name}_peak"] = float(np.max(ts))
        features[f"{dim_name}_std"]  = float(np.std(ts))

    return features


def extract_attention_features(preds: np.ndarray) -> dict[str, float]:
    """
    Attention / hook features — based on temporal dynamics of GLOBAL
    activation (not a specific region group).

    From Creative_Quality_Scorer_Design.md §3.4:
      1. Global mean activation per second
      2. First-3-second "hook" ratio vs video average
      3. Onset speed (first second above threshold)
    """
    features = {}
    n_seconds = preds.shape[0]

    # Global activation per second (mean across all vertices)
    global_ts = np.mean(preds, axis=1)  # shape: (n_seconds,)

    video_mean = float(np.mean(global_ts))
    video_std  = float(np.std(global_ts))

    # First 3 seconds mean
    first3 = global_ts[:min(3, n_seconds)]
    first3_mean = float(np.mean(first3))

    # Hook ratio: how much stronger are the first 3s vs the video average
    if abs(video_mean) > 1e-9:
        hook_ratio = first3_mean / video_mean
    else:
        hook_ratio = 1.0

    # Onset speed: first second where activation exceeds mean + 0.5*std
    threshold = video_mean + 0.5 * video_std
    above = np.where(global_ts > threshold)[0]
    onset_second = float(above[0]) if len(above) > 0 else float(n_seconds)

    features["attention_hook_ratio"]    = hook_ratio
    features["attention_onset_second"]  = onset_second
    features["attention_first3s_mean"]  = first3_mean

    return features


def extract_temporal_features(preds: np.ndarray) -> dict[str, float]:
    """
    Temporal dynamics features derived from the global activation time-series.
    """
    features = {}
    n_seconds = preds.shape[0]

    # Global activation per second
    global_ts = np.mean(preds, axis=1)

    video_mean = float(np.mean(global_ts))
    video_std  = float(np.std(global_ts))

    # Peak second
    features["peak_second"] = float(np.argmax(global_ts))

    # Engagement variance
    features["engagement_variance"] = video_std

    # Engagement slopes (linear fit on first-half and second-half)
    mid = n_seconds // 2
    if mid >= 2:
        x_first = np.arange(mid)
        slope_first = np.polyfit(x_first, global_ts[:mid], 1)[0]
        features["engagement_slope_first_half"] = float(slope_first)

        x_second = np.arange(n_seconds - mid)
        slope_second = np.polyfit(x_second, global_ts[mid:], 1)[0]
        features["engagement_slope_second_half"] = float(slope_second)
    else:
        features["engagement_slope_first_half"]  = 0.0
        features["engagement_slope_second_half"] = 0.0

    # Number of engagement peaks (local maxima above 1σ)
    if video_std > 1e-9:
        prominence = video_std * 0.5
        peaks, _ = find_peaks(global_ts, prominence=prominence)
        features["engagement_peak_count"] = float(len(peaks))
    else:
        features["engagement_peak_count"] = 0.0

    # Longest sustained engagement above mean
    above_mean = global_ts > video_mean
    longest_run = 0
    current_run = 0
    for val in above_mean:
        if val:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    features["longest_sustained_above_mean"] = float(longest_run)

    # Percentage of video above mean activation
    features["pct_above_mean"] = float(np.sum(above_mean) / n_seconds)

    # Global activation range
    features["global_activation_range"] = float(np.max(global_ts) - np.min(global_ts))

    return features


def extract_global_features(preds: np.ndarray) -> dict[str, float]:
    """Global summary statistics across ALL vertices and seconds."""
    features = {}

    features["total_neural_energy"]    = float(np.sum(np.maximum(preds, 0)))
    features["global_mean_activation"] = float(np.mean(preds))
    features["global_peak_activation"] = float(np.max(preds))
    features["video_duration_seconds"] = float(preds.shape[0])
    features["n_vertices"]             = float(preds.shape[1])

    return features


def extract_individual_region_features(
    preds: np.ndarray,
    atlas: DestrieuxAtlas,
) -> dict[str, float]:
    """Per-region mean and peak for the most discriminative individual regions."""
    features = {}

    for short_name, atlas_label in KEY_INDIVIDUAL_REGIONS.items():
        verts = atlas.vertices_for(atlas_label)
        if len(verts) == 0:
            features[f"{short_name}_mean"] = np.nan
            features[f"{short_name}_peak"] = np.nan
            continue

        ts = np.mean(preds[:, verts], axis=1)
        features[f"{short_name}_mean"] = float(np.mean(ts))
        features[f"{short_name}_peak"] = float(np.max(ts))

    return features


def compute_weighted_scores(features: dict[str, float]) -> dict[str, float]:
    """
    Compute 0-100 normalised dimension scores and the overall weighted score.

    Normalization note: Without a calibration baseline, raw scores are NOT
    on a 0-100 scale yet.  We store the raw weighted values here.  True
    0-100 normalization requires min/max from the full calibration set,
    which is computed in a post-processing step after ALL videos are
    extracted.  For now, we store raw means which can be normalized later.
    """
    scores = {}

    # Raw dimension scores (mean activation for each group)
    dim_keys = {
        "visual":    "visual_mean",
        "auditory":  "auditory_mean",
        "emotional": "emotional_mean",
        "language":  "language_mean",
        "attention": "attention_first3s_mean",
    }

    overall = 0.0
    for dim_name, feat_key in dim_keys.items():
        raw = features.get(feat_key, 0.0)
        scores[f"{dim_name}_score_raw"] = raw
        overall += raw * DIMENSION_WEIGHTS[dim_name]

    scores["overall_engagement_score_raw"] = overall

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# PER-FILE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_npz(
    npz_path: Path,
    atlas: DestrieuxAtlas,
) -> dict[str, float]:
    """
    Load a single .npz file and return a flat dict of all features.
    """
    data = np.load(npz_path, allow_pickle=True)
    preds = data["preds"]  # shape: (n_seconds, 20484)

    log.info(
        "  Processing %s  shape=%s",
        npz_path.name, preds.shape,
    )

    features: dict[str, float] = {}

    # 1. Per-dimension features (visual, auditory, emotional, language)
    features.update(extract_dimension_features(preds, atlas))

    # 2. Attention / hook features
    features.update(extract_attention_features(preds))

    # 3. Temporal dynamics
    features.update(extract_temporal_features(preds))

    # 4. Global stats
    features.update(extract_global_features(preds))

    # 5. Key individual regions
    features.update(extract_individual_region_features(preds, atlas))

    # 6. Weighted overall scores
    features.update(compute_weighted_scores(features))

    return features


def extract_engagement_curve(
    npz_path: Path,
    atlas: DestrieuxAtlas,
) -> pd.DataFrame:
    """
    Second-by-second engagement curve for visualization.
    Returns a DataFrame with columns: second, global, visual, auditory,
    emotional, language.
    """
    data = np.load(npz_path, allow_pickle=True)
    preds = data["preds"]
    n_seconds = preds.shape[0]

    curve = {"second": np.arange(n_seconds)}
    curve["global"] = np.mean(preds, axis=1)

    for dim_name, region_labels in REGION_GROUPS.items():
        verts = atlas.vertices_for_group(region_labels)
        if len(verts) > 0:
            curve[dim_name] = np.mean(preds[:, verts], axis=1)
        else:
            curve[dim_name] = np.zeros(n_seconds)

    return pd.DataFrame(curve)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def discover_npz_files(root_dir: Path) -> list[Path]:
    """Recursively find all .npz files under root_dir."""
    files = sorted(root_dir.rglob("*.npz"))
    log.info("Discovered %d .npz file(s) under %s", len(files), root_dir)
    return files


def run_batch(
    npz_files: list[Path],
    atlas: DestrieuxAtlas,
    output_csv: Path,
    save_curves: bool = False,
    curves_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Process all .npz files and write the feature CSV."""
    rows = []

    for idx, npz_path in enumerate(npz_files, 1):
        log.info("━━━ [%d/%d] %s ━━━", idx, len(npz_files), npz_path.name)

        try:
            features = extract_features_from_npz(npz_path, atlas)
        except Exception as exc:
            log.error("❌  Failed to process %s: %s", npz_path.name, exc)
            continue

        # Metadata columns
        features["filename"] = npz_path.stem  # e.g. "CFHO_Boat_V1_720x900_30s"
        features["video_folder"] = npz_path.parent.name

        rows.append(features)

        # Optional: save per-video engagement curve
        if save_curves:
            try:
                curve_df = extract_engagement_curve(npz_path, atlas)
                dest = (curves_dir or npz_path.parent) / f"{npz_path.stem}_curve.csv"
                curve_df.to_csv(dest, index=False)
                log.info("  📈  Saved engagement curve → %s", dest.name)
            except Exception as exc:
                log.warning("  ⚠️  Could not save curve for %s: %s", npz_path.name, exc)

    if not rows:
        log.error("No files were successfully processed.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Move metadata columns to front
    meta_cols = ["filename", "video_folder"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    df.to_csv(output_csv, index=False)
    log.info("━" * 60)
    log.info("✅  Wrote %d rows × %d columns → %s", len(df), len(df.columns), output_csv)
    log.info("━" * 60)

    return df


def print_feature_summary(df: pd.DataFrame):
    """Print a summary of the extracted features."""
    log.info("")
    log.info("FEATURE SUMMARY")
    log.info("━" * 60)

    # Skip metadata columns
    feat_cols = [c for c in df.columns if c not in ("filename", "video_folder")]
    log.info("  Feature columns: %d", len(feat_cols))
    log.info("  Videos processed: %d", len(df))

    # Check for NaNs
    nan_counts = df[feat_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        log.warning("  ⚠️  Columns with NaN values:")
        for col, count in nan_cols.items():
            log.warning("    %s: %d NaN(s)", col, count)
    else:
        log.info("  ✅  No NaN values in any feature column")

    # Print a few key stats
    for col in ["overall_engagement_score_raw", "visual_mean", "emotional_mean",
                "attention_hook_ratio", "peak_second"]:
        if col in df.columns:
            log.info(
                "  %-35s  min=%.4f  max=%.4f  mean=%.4f",
                col, df[col].min(), df[col].max(), df[col].mean(),
            )

    log.info("━" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract brain features from TRIBEv2 .npz files → CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--npz-dir",
        help="Directory containing .npz files (searched recursively).",
    )
    group.add_argument(
        "--video-dir",
        help="Directory containing videos + their .npz files "
             "(e.g. the Marketing Final Videos folder).",
    )

    parser.add_argument(
        "--output", "-o",
        default="brain_features.csv",
        help="Output CSV file path (default: brain_features.csv).",
    )
    parser.add_argument(
        "--save-curves",
        action="store_true",
        help="Save per-video engagement curves as separate CSV files.",
    )
    parser.add_argument(
        "--curves-dir",
        default=None,
        help="Directory for engagement curve CSVs (default: same as .npz file).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine root directory
    root = Path(args.npz_dir or args.video_dir)
    if not root.is_dir():
        log.error("Directory does not exist: %s", root)
        sys.exit(1)

    # Discover .npz files
    npz_files = discover_npz_files(root)
    if not npz_files:
        log.error("No .npz files found under %s", root)
        sys.exit(1)

    # Load atlas (once)
    atlas = DestrieuxAtlas()

    # Curves directory
    curves_dir = Path(args.curves_dir) if args.curves_dir else None
    if curves_dir:
        curves_dir.mkdir(parents=True, exist_ok=True)

    # Run extraction
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_batch(
        npz_files=npz_files,
        atlas=atlas,
        output_csv=output_path,
        save_curves=args.save_curves,
        curves_dir=curves_dir,
    )

    print_feature_summary(df)


if __name__ == "__main__":
    main()
