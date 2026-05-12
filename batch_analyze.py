"""
TRIBEv2 Batch Video Analyzer
=============================

Scans a directory tree for .mp4 video files, submits each to the TRIBEv2
inference API, polls until completion, and saves the resulting .npz brain-
activity file next to the original video with the same base name.

State is persisted to batch_state.json so the script can safely resume after
being interrupted — in-progress pod jobs are reattached, not re-submitted.

Usage:
    python batch_analyze.py --base-url <API_BASE_URL>

Example:
    python batch_analyze.py --base-url https://abc123-8000.proxy.runpod.net

Environment variables (optional overrides):
    TRIBEV2_API_BASE_URL  — API base URL (instead of --base-url)
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("batch_analyze")

# ── Constants ────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS       = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
POLL_INTERVAL_SECONDS  = 30       # how often to check job status
MAX_POLL_DURATION_SECONDS = 3600  # give up after 1 hour per video
TERMINAL_STATUSES      = {"COMPLETED", "FAILED", "DELETED"}

# Default video directory
DEFAULT_VIDEO_DIR = str(
    Path(r"D:\Work\R&D\tribev2\test_videos")
    / "applovin_videos_2"
    # / "Marketing Final Videos-20260428T053804Z-3-003"
    # / "Marketing Final Videos"
)

# State file path (written alongside this script)
STATE_FILE = Path(__file__).parent / "batch_state.json"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VideoState:
    """Persistent state for a single video's job."""
    video_path: str          # absolute path (str for JSON serialisation)
    npz_path: str
    job_id: Optional[str] = None
    status: str = "NOT_SUBMITTED"   # NOT_SUBMITTED | SUBMITTED | COMPLETED | FAILED | SKIPPED
    error: Optional[str] = None


@dataclass
class BatchResult:
    total: int = 0
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list = field(default_factory=list)


# ── State persistence ────────────────────────────────────────────────────────

class StateStore:
    """
    Loads and saves batch_state.json.
    Key: absolute video path string → VideoState dict.
    """

    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
            log.info("Loaded state from %s  (%d entries)", self.path, len(self._data))
        else:
            log.info("No existing state file — starting fresh.")

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, video_path: str) -> Optional[VideoState]:
        d = self._data.get(video_path)
        if d:
            return VideoState(**d)
        return None

    def upsert(self, vs: VideoState):
        self._data[vs.video_path] = asdict(vs)
        self.save()

    def all_states(self) -> list[VideoState]:
        return [VideoState(**d) for d in self._data.values()]


# ── API Client ───────────────────────────────────────────────────────────────

class TRIBEv2Client:
    """Minimal HTTP client for the TRIBEv2 inference API."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def submit_video(self, video_path: Path) -> str:
        url = f"{self.base_url}/api/v1/jobs/analyze"
        log.info("Uploading  %s", video_path.name)
        with open(video_path, "rb") as f:
            resp = self.session.post(
                url,
                files={"video": (video_path.name, f, "video/mp4")},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        log.info("Submitted  %s  →  job_id=%s", video_path.name, job_id)
        return job_id

    def get_status(self, job_id: str) -> dict:
        url = f"{self.base_url}/api/v1/jobs/{job_id}/status"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download_result(self, job_id: str, dest: Path) -> Path:
        url = f"{self.base_url}/api/v1/jobs/{job_id}/result"
        log.info("Downloading result  job=%s  →  %s", job_id, dest.name)
        with self.session.get(url, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        log.info("Saved  %s  (%d bytes)", dest.name, dest.stat().st_size)
        return dest


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_videos(root_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in root_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
    )
    log.info("Discovered %d video(s) under %s", len(videos), root_dir)
    return videos


def npz_path_for(video_path: Path) -> Path:
    return video_path.with_suffix(".npz")


# ── Polling ──────────────────────────────────────────────────────────────────

def poll_until_done(
    client: TRIBEv2Client,
    vs: VideoState,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    max_duration: int = MAX_POLL_DURATION_SECONDS,
    store: Optional[StateStore] = None,
) -> str:
    """Poll the API until vs.job_id reaches a terminal state."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > max_duration:
            vs.status = "TIMEOUT"
            vs.error = f"Timed out after {max_duration}s"
            if store:
                store.upsert(vs)
            log.warning("⏰  Timeout  %s  (job=%s)", Path(vs.video_path).name, vs.job_id)
            return vs.status

        data = client.get_status(vs.job_id)
        api_status = data.get("status", "UNKNOWN")
        vs.error = data.get("error_message")

        if api_status in TERMINAL_STATUSES:
            vs.status = api_status
            if store:
                store.upsert(vs)
            return vs.status

        log.info(
            "⏳  %s  pod_status=%s  elapsed=%.0fs  (retry in %ds)",
            Path(vs.video_path).name, api_status, elapsed, poll_interval,
        )
        time.sleep(poll_interval)


# ── Per-video logic ──────────────────────────────────────────────────────────

def process_video(
    client: TRIBEv2Client,
    video_path: Path,
    store: StateStore,
    skip_existing: bool = True,
) -> VideoState:
    """
    Resume-aware: reloads state from store so interrupted runs pick up where
    they left off instead of re-submitting to the pod.
    """
    key = str(video_path)
    npz_dest = npz_path_for(video_path)

    # ── Already have the .npz on disk ────────────────────────────────────
    if skip_existing and npz_dest.exists():
        log.info("⏭  Skipping %s  (.npz already on disk)", video_path.name)
        vs = VideoState(video_path=key, npz_path=str(npz_dest), status="SKIPPED")
        store.upsert(vs)
        return vs

    # ── Load or create state ──────────────────────────────────────────────
    vs = store.get(key) or VideoState(video_path=key, npz_path=str(npz_dest))

    # ── Already completed in a previous run ───────────────────────────────
    if vs.status == "COMPLETED" and npz_dest.exists():
        log.info("⏭  Already completed in prior run: %s", video_path.name)
        return vs

    # ── Has a job_id → reattach to existing pod job ───────────────────────
    if vs.job_id:
        log.info(
            "♻️  Reattaching  %s  →  job_id=%s",
            video_path.name, vs.job_id,
        )
        # Check current pod status first
        try:
            data = client.get_status(vs.job_id)
            pod_status = data.get("status", "UNKNOWN")
        except Exception as exc:
            log.error("Could not fetch status for job %s: %s — will re-submit", vs.job_id, exc)
            vs.job_id = None  # fall through to re-submit

        if vs.job_id:
            if pod_status in TERMINAL_STATUSES:
                vs.status = pod_status
                vs.error = data.get("error_message")
                store.upsert(vs)
            else:
                # Still running — continue polling
                log.info("  pod_status=%s  continuing to poll…", pod_status)
                poll_until_done(client, vs, store=store)
    else:
        # ── Fresh submit ──────────────────────────────────────────────────
        try:
            vs.job_id = client.submit_video(video_path)
            vs.status = "SUBMITTED"
            store.upsert(vs)
        except Exception as exc:
            vs.status = "SUBMIT_ERROR"
            vs.error = str(exc)
            log.error("❌  Submit failed  %s: %s", video_path.name, exc)
            store.upsert(vs)
            return vs

        poll_until_done(client, vs, store=store)

    # ── Download if completed ─────────────────────────────────────────────
    if vs.status == "COMPLETED":
        try:
            client.download_result(vs.job_id, npz_dest)
            log.info("✅  %s", npz_dest.name)
        except Exception as exc:
            vs.status = "DOWNLOAD_ERROR"
            vs.error = str(exc)
            log.error("❌  Download failed  %s: %s", video_path.name, exc)
            store.upsert(vs)
    else:
        log.warning(
            "⚠️  %s  finished with status=%s  error=%s",
            video_path.name, vs.status, vs.error,
        )

    return vs


# ── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(
    client: TRIBEv2Client,
    video_dir: Path,
    store: StateStore,
    skip_existing: bool = True,
) -> BatchResult:
    videos = discover_videos(video_dir)
    result = BatchResult(total=len(videos))

    for idx, vp in enumerate(videos, 1):
        log.info("━━━ [%d/%d] %s ━━━", idx, result.total, vp.name)
        vs = process_video(client, vp, store, skip_existing=skip_existing)

        if vs.status == "SKIPPED":
            result.skipped += 1
        elif vs.status == "COMPLETED":
            result.submitted += 1
            result.completed += 1
        else:
            result.submitted += 1
            result.failed += 1
            result.errors.append({
                "file": vs.video_path,
                "status": vs.status,
                "error": vs.error,
            })

    return result


def print_summary(result: BatchResult):
    log.info("━" * 60)
    log.info("BATCH SUMMARY")
    log.info("━" * 60)
    log.info("  Total videos found : %d", result.total)
    log.info("  Skipped (existing) : %d", result.skipped)
    log.info("  Submitted          : %d", result.submitted)
    log.info("  Completed ✅       : %d", result.completed)
    log.info("  Failed    ❌       : %d", result.failed)
    if result.errors:
        log.info("  Errors:")
        for err in result.errors:
            log.info("    %s  →  %s: %s",
                     Path(err["file"]).name, err["status"], err["error"])
    log.info("━" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-analyze videos via TRIBEv2 inference API (resume-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("TRIBEV2_API_BASE_URL", ""),
        help="TRIBEv2 API base URL.  Also: TRIBEV2_API_BASE_URL env var.",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        help="Root directory containing video files.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process videos even if .npz already exists on disk.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between status polls (default: {POLL_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--max-poll",
        type=int,
        default=MAX_POLL_DURATION_SECONDS,
        help=f"Max seconds to wait per video (default: {MAX_POLL_DURATION_SECONDS}).",
    )
    parser.add_argument(
        "--state-file",
        default=str(STATE_FILE),
        help=f"Path to resume-state JSON (default: {STATE_FILE}).",
    )
    parser.add_argument(
        "--save-report",
        default="",
        help="If set, write a JSON summary report to this path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.base_url:
        log.error("No API base URL. Use --base-url or set TRIBEV2_API_BASE_URL.")
        sys.exit(1)

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        log.error("Video directory does not exist: %s", video_dir)
        sys.exit(1)

    # Apply CLI overrides to polling constants
    global POLL_INTERVAL_SECONDS, MAX_POLL_DURATION_SECONDS
    POLL_INTERVAL_SECONDS     = args.poll_interval
    MAX_POLL_DURATION_SECONDS = args.max_poll

    store  = StateStore(Path(args.state_file))
    client = TRIBEv2Client(base_url=args.base_url)

    log.info("API base URL  : %s", args.base_url)
    log.info("Video dir     : %s", video_dir)
    log.info("State file    : %s", args.state_file)
    log.info("Skip existing : %s", not args.no_skip)

    # Graceful Ctrl+C — state is already saved per-video, so just exit cleanly
    def _sigint(sig, frame):
        log.warning("\n⚠️  Interrupted — progress saved to %s", args.state_file)
        log.warning("Re-run the same command to resume.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    result = run_batch(client, video_dir, store, skip_existing=not args.no_skip)
    print_summary(result)

    if args.save_report:
        rp = Path(args.save_report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f:
            json.dump(
                {
                    "total": result.total,
                    "submitted": result.submitted,
                    "completed": result.completed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "errors": result.errors,
                },
                f, indent=2,
            )
        log.info("Report saved → %s", rp)


if __name__ == "__main__":
    main()
