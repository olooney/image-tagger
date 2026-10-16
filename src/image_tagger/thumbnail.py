import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .compare import EmbeddingCache, _load_embedding_cache, _save_embedding_cache
from .constants import (
    CLIP_MODEL,
    DEDUPE_EMBEDDINGS_FILENAME,
    GALLERY_NAME,
    IMAGE_EXTENSIONS,
    METADATA_FILENAME,
)
from .util import Pathish, display_file_operation, display_path
from .wall import image_aspect_ratios, infer_wall_layout

register_heif_opener()

DEFAULT_MAX_WIDTH: int = 400
DEFAULT_MAX_HEIGHT: int = 600
PUBLISH_FILE_EXTENSIONS: set[str] = {".csv", ".npz"}
PUBLISH_FILENAMES: set[str] = {GALLERY_NAME.name}


def _display_operation(
    action: str,
    source: Path,
    target: Path,
    *,
    verbose: int,
    relative_to: Path,
) -> None:
    """Print a file operation before it runs."""
    if verbose >= 2:
        print(
            display_file_operation(
                action,
                source,
                target,
                verbose=verbose,
                relative_to=relative_to,
            ),
            end="",
        )


def _display_outcome(*, succeeded: bool, verbose: int) -> None:
    """Print one compact or detailed file-operation outcome."""
    if verbose == 1:
        print(".", end="", flush=True)
    elif verbose >= 2:
        print("success!" if succeeded else "error!")


def _display_skip(
    source: Path,
    target: Path,
    *,
    verbose: int,
    relative_to: Path,
) -> None:
    """Print one detailed skipped-file line."""
    if verbose >= 2:
        operation = display_file_operation(
            "skipping",
            source,
            target,
            verbose=verbose,
            relative_to=relative_to,
        )
        print(operation.removesuffix(" ..."))


def _display_removal(
    path: Path,
    *,
    verbose: int,
    relative_to: Path,
) -> None:
    """Print one thumbnail removal before it runs."""
    if verbose >= 2:
        print(
            f"removing {display_path(path, verbose=verbose, relative_to=relative_to)} ...",
            end="",
        )


def _format_summary(
    success_count: int,
    metadata_count: int,
    error_count: int,
    skip_count: int,
    removal_count: int,
) -> str:
    """Format nonzero thumbnail operation counts."""
    counts: list[str] = []
    if success_count:
        counts.append(f"{success_count} succeeded")
    if metadata_count:
        counts.append(f"{metadata_count} metadata")
    if error_count:
        error_label = "error" if error_count == 1 else "errors"
        counts.append(f"{error_count} {error_label}")
    if skip_count:
        counts.append(f"{skip_count} skipped")
    if removal_count:
        counts.append(f"{removal_count} removed")
    return ", ".join(counts)


def _translate_thumbnail_metadata(
    source_path: Path,
    destination_path: Path,
    destination_paths_by_source: dict[Path, Path],
) -> None:
    """Point copied metadata rows at their generated thumbnails."""
    with source_path.open(newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None or "original_filepath" not in reader.fieldnames:
            return
        rows = list(reader)

    for row in rows:
        original_filepath = row.get("original_filepath", "").strip()
        if not original_filepath:
            continue
        original_path = Path(original_filepath)
        candidates = [original_path]
        clean_filename = row.get("clean_filename", "").strip()
        if clean_filename:
            candidates.append(original_path.with_name(clean_filename))
        target = next(
            (
                destination_paths_by_source.get(candidate.resolve())
                for candidate in candidates
                if candidate.resolve() in destination_paths_by_source
            ),
            None,
        )
        if target is None or not target.is_file():
            continue
        row["original_filepath"] = os.fspath(target.resolve())
        if "original_filename" in row:
            row["original_filename"] = target.name

    with destination_path.open("w", newline="", encoding="utf-8") as destination_file:
        writer = csv.DictWriter(destination_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _translate_thumbnail_cache(
    source_path: Path,
    destination_path: Path,
    destination_paths_by_source: dict[Path, Path],
) -> None:
    """Re-key fresh source embeddings and review state for thumbnails."""
    source_cache = _load_embedding_cache(source_path, CLIP_MODEL)
    entries: dict[str, tuple[int, int, np.ndarray]] = {}
    reviewed_thresholds: dict[str, tuple[float, float]] = {}
    for source_image, destination_image in destination_paths_by_source.items():
        source_key = str(source_image)
        source_entry = source_cache.entries.get(source_key)
        if source_entry is None or not source_image.is_file():
            continue
        source_size, source_mtime_ns, vector = source_entry
        source_stat = source_image.stat()
        if (source_size, source_mtime_ns) != (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ) or not destination_image.is_file():
            continue
        destination_stat = destination_image.stat()
        destination_key = str(destination_image.resolve())
        entries[destination_key] = (
            destination_stat.st_size,
            destination_stat.st_mtime_ns,
            vector,
        )
        if thresholds := source_cache.reviewed_thresholds.get(source_key):
            reviewed_thresholds[destination_key] = thresholds

    if entries:
        _save_embedding_cache(
            destination_path,
            CLIP_MODEL,
            EmbeddingCache(
                entries=entries,
                reviewed_thresholds=reviewed_thresholds,
            ),
        )


def create_thumbnails(
    source: Pathish,
    destination: Pathish,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    dry_run: bool = False,
    verbose: int = 1,
) -> None:
    """Create a size-limited copy of an image collection for publishing."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Thumbnail destination must not be inside the source directory.")

    source_paths = [path for path in source.rglob("*") if path.is_file()]
    image_paths = [
        path for path in source_paths if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    destination_paths_by_source = {
        image_path.resolve(): destination / image_path.relative_to(source)
        for image_path in image_paths
    }
    expected_destination_paths = {
        path.resolve() for path in destination_paths_by_source.values()
    }
    removed_source_thumbnail_paths = [
        path
        for path in destination.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve() not in expected_destination_paths
    ]
    source_gallery = source / GALLERY_NAME
    source_gallery_is_current = source_gallery.is_file() and (
        not image_paths
        or source_gallery.stat().st_mtime_ns
        >= max(path.stat().st_mtime_ns for path in image_paths)
    )
    newer_thumbnail_paths = {
        image_path
        for image_path in image_paths
        if (destination / image_path.relative_to(source)).is_file()
        and (destination / image_path.relative_to(source)).stat().st_mtime_ns
        > image_path.stat().st_mtime_ns
    }
    aspect_ratios: dict[Path, float] = {}
    for image_path in image_paths:
        aspect_path = (
            destination / image_path.relative_to(source)
            if image_path in newer_thumbnail_paths
            else image_path
        )
        try:
            measured_ratios = image_aspect_ratios([aspect_path])
        except OSError:
            continue
        if measured_ratios:
            aspect_ratios[image_path] = next(iter(measured_ratios.values()))
    _, double_wide_paths = infer_wall_layout(aspect_ratios)

    success_count = 0
    metadata_count = 0
    error_count = 0
    skip_count = 0
    removal_count = 0
    try:
        for source_path in source_paths:
            relative_path = source_path.relative_to(source)
            destination_path = destination / relative_path
            suffix = source_path.suffix.lower()

            if suffix in IMAGE_EXTENSIONS:
                if source_path in newer_thumbnail_paths:
                    _display_skip(
                        source_path,
                        destination_path,
                        verbose=verbose,
                        relative_to=source,
                    )
                    skip_count += 1
                    continue
                _display_operation(
                    "thumbnailing",
                    source_path,
                    destination_path,
                    verbose=verbose,
                    relative_to=source,
                )
                try:
                    if not dry_run:
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        with Image.open(source_path) as image:
                            image_format = image.format
                            output = ImageOps.exif_transpose(image)
                            output_max_width = (
                                max_width * 2
                                if source_path in double_wide_paths
                                else max_width
                            )
                            output.thumbnail(
                                (output_max_width, max_height),
                                Image.Resampling.LANCZOS,
                            )
                            output.save(destination_path, format=image_format)
                    success_count += 1
                    _display_outcome(succeeded=True, verbose=verbose)
                except Exception:
                    error_count += 1
                    _display_outcome(succeeded=False, verbose=verbose)
                    raise
            elif (
                suffix in PUBLISH_FILE_EXTENSIONS
                or source_path.name in PUBLISH_FILENAMES
            ):
                _display_operation(
                    "copying",
                    source_path,
                    destination_path,
                    verbose=verbose,
                    relative_to=source,
                )
                try:
                    if not dry_run:
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, destination_path)
                    metadata_count += 1
                    if verbose >= 2:
                        _display_outcome(succeeded=True, verbose=verbose)
                except Exception:
                    error_count += 1
                    _display_outcome(succeeded=False, verbose=verbose)
                    raise
        for thumbnail_path in removed_source_thumbnail_paths:
            _display_removal(
                thumbnail_path,
                verbose=verbose,
                relative_to=destination,
            )
            try:
                if not dry_run:
                    thumbnail_path.unlink()
                removal_count += 1
                _display_outcome(succeeded=True, verbose=verbose)
            except Exception:
                error_count += 1
                _display_outcome(succeeded=False, verbose=verbose)
                raise
        if not dry_run:
            metadata_source = source / METADATA_FILENAME
            metadata_destination = destination / METADATA_FILENAME
            if metadata_source.is_file() and metadata_destination.is_file():
                _translate_thumbnail_metadata(
                    metadata_source,
                    metadata_destination,
                    destination_paths_by_source,
                )
            cache_source = source / DEDUPE_EMBEDDINGS_FILENAME
            cache_destination = destination / DEDUPE_EMBEDDINGS_FILENAME
            if cache_source.is_file() and cache_destination.is_file():
                _translate_thumbnail_cache(
                    cache_source,
                    cache_destination,
                    destination_paths_by_source,
                )
            destination_gallery = destination / GALLERY_NAME
            destination_images = [
                path
                for path in destination_paths_by_source.values()
                if path.is_file()
            ]
            if (
                source_gallery_is_current
                and destination_gallery.is_file()
                and destination_images
            ):
                gallery_stat = destination_gallery.stat()
                latest_image_mtime = max(
                    path.stat().st_mtime_ns for path in destination_images
                )
                if gallery_stat.st_mtime_ns < latest_image_mtime:
                    os.utime(
                        destination_gallery,
                        ns=(gallery_stat.st_atime_ns, latest_image_mtime),
                    )
    finally:
        summary = _format_summary(
            success_count,
            metadata_count,
            error_count,
            skip_count,
            removal_count,
        )
        if verbose >= 1 and summary:
            if verbose == 1 and success_count + error_count + removal_count:
                print()
            print(summary)