#!/usr/bin/env python
"""Download the official NEU-DET archive and optionally prepare YOLO labels."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FILE_ID = "1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k"
OFFICIAL_PAGE = "https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm"
DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={OFFICIAL_FILE_ID}&export=download&confirm=t"
)


def _progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        print(f"\rDownloading NEU-DET: {percent:5.1f}%", end="", flush=True)


def download(destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000 and not force:
        print(f"Using existing archive: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, temporary, reporthook=_progress)
        print()
        if temporary.stat().st_size < 1_000_000 or not zipfile.is_zipfile(temporary):
            raise RuntimeError("Downloaded file is not a valid NEU-DET ZIP archive")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract(archive: Path, destination: Path, force: bool = False) -> Path:
    marker = destination / "NEU-DET" / "ANNOTATIONS"
    if marker.exists() and not force:
        print(f"Using existing extracted dataset: {destination}")
        return destination
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    print(f"Extracted dataset to: {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "downloads" / "NEU-DET.zip",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw" / "neu_det",
    )
    parser.add_argument("--force", action="store_true", help="download and extract again")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="also convert VOC XML to a stratified YOLO split",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"Official dataset page: {OFFICIAL_PAGE}")
    try:
        archive = download(args.archive.resolve(), force=args.force)
        raw_dir = extract(archive, args.raw_dir.resolve(), force=args.force)
        if args.prepare:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from prepare_dataset import prepare_neu_det

            prepare_neu_det(
                input_dir=raw_dir,
                output_dir=PROJECT_ROOT / "datasets" / "neu_det",
                seed=42,
                ratios=(0.7, 0.15, 0.15),
                overwrite=args.force,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Manual download: {OFFICIAL_PAGE}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

