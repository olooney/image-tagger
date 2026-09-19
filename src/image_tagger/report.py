import csv
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from .compare import _reviewed_cached_image_paths, image_dimensions
from .constants import (
    CLIP_MODEL,
    DEDUPE_EMBEDDINGS_FILENAME,
    DEFAULT_AUTOMATIC_THRESHOLD,
    DEFAULT_LARGE_IMAGE_THRESHOLD,
    DEFAULT_LLM_THRESHOLD,
    GALLERY_NAME,
    IMAGE_EXTENSIONS,
)
from .tagging import find_images
from .util import Pathish, quote_display_path


def _format_report_counts(title: str, counts: Counter[str]) -> list[str]:
    """Format one labeled, right-aligned report breakdown."""
    if not counts:
        return []
    longest_name = max(len(name) for name in counts)
    widest_count = len(str(max(counts.values())))
    lines = [f"{title}:"]
    lines.extend(
        f"    {name:<{longest_name}}     {count:>{widest_count}}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return lines


def _format_file_size(size_bytes: int) -> str:
    """Format a file size using decimal units."""
    units = ["B", "Kb", "Mb", "Gb", "Tb"]
    size = float(size_bytes)
    for unit in units:
        if size < 1000 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1000
    raise AssertionError("unreachable")


def _metadata_paths(row: Mapping[str, str]) -> set[Path]:
    """Return possible current image paths represented by one metadata row."""
    original_filepath = row.get("original_filepath", "").strip()
    if not original_filepath:
        return set()
    original_path = Path(original_filepath)
    paths = {original_path}
    clean_filename = row.get("clean_filename", "").strip()
    if clean_filename:
        paths.add(original_path.with_name(clean_filename))
    return paths


def _format_largest_images(
    directory_path: Path,
    image_paths: set[Path],
) -> list[str]:
    """Format the largest image paths as an aligned report table."""
    sorted_paths = sorted(
        image_paths,
        key=lambda path: (
            -image_dimensions(path)[1],
            -image_dimensions(path)[0],
            -path.stat().st_size,
            path,
        ),
    )
    items = [
        (
            quote_display_path(path.relative_to(directory_path)),
            f"{image_dimensions(path)[0]}x{image_dimensions(path)[1]}",
            _format_file_size(path.stat().st_size),
        )
        for path in sorted_paths
    ]
    filename_width = max(len(filename) for filename, _, _ in items)
    dimensions_width = max(len(dimensions) for _, dimensions, _ in items)
    size_width = max(len(size) for _, _, size in items)
    return [
        f"{filename:<{filename_width}}     {dimensions:>{dimensions_width}}     "
        f"{size:>{size_width}}"
        for filename, dimensions, size in items
    ]


def _format_filename_mismatches(
    directory_path: Path,
    mismatches: list[tuple[Path, str]],
) -> list[str]:
    """Format current and suggested filenames in aligned columns."""
    items = [
        (quote_display_path(path.relative_to(directory_path)), clean_filename)
        for path, clean_filename in mismatches[:5]
    ]
    filename_width = max(len(current_filename) for current_filename, _ in items)
    return [
        f"{current_filename:<{filename_width}}     {clean_filename}"
        for current_filename, clean_filename in items
    ]


def gallery_is_out_of_date(directory_path: Path, image_paths: list[Path]) -> bool:
    """Return whether the gallery index predates the newest image."""
    gallery_path = directory_path / GALLERY_NAME
    return (
        bool(image_paths)
        and gallery_path.is_file()
        and (
            gallery_path.stat().st_mtime_ns
            < max(path.stat().st_mtime_ns for path in image_paths)
        )
    )


def report_images(
    directory: Pathish,
    metadata_filename: Pathish,
    large_image_threshold: int = DEFAULT_LARGE_IMAGE_THRESHOLD,
) -> str:
    """Build a text report for the images in one directory tree."""
    directory_path = Path(directory)
    image_paths = sorted(
        path
        for path in directory_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    metadata_by_path: dict[Path, dict[str, str]] = {}
    metadata_path = Path(metadata_filename)
    if metadata_path.is_file():
        with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
            for row in csv.DictReader(metadata_file):
                for path in _metadata_paths(row):
                    metadata_by_path[path.resolve()] = row

    extension_counts = Counter(path.suffix.lower() for path in image_paths)
    category_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    missing_metadata_count = 0
    clean_filename_mismatches: list[tuple[Path, str]] = []
    for path in image_paths:
        row = metadata_by_path.get(path.resolve())
        if row is None:
            missing_metadata_count += 1
            continue
        category = row.get("category", "").strip()
        genre = row.get("genre", "").strip()
        if category:
            category_counts[category] += 1
        if genre:
            genre_counts[genre] += 1
        try:
            tags = json.loads(row.get("tags", "[]"))
        except json.JSONDecodeError:
            tags = []
        if isinstance(tags, list):
            tag_counts.update(tag for tag in tags if isinstance(tag, str) and tag)
        clean_name = row.get("clean_filename", "").strip()
        if clean_name and path.name != clean_name:
            clean_filename_mismatches.append((path, clean_name))

    dedupe_image_paths = find_images(directory_path)
    reviewed_paths = _reviewed_cached_image_paths(
        dedupe_image_paths,
        directory_path / DEDUPE_EMBEDDINGS_FILENAME,
        CLIP_MODEL,
        DEFAULT_AUTOMATIC_THRESHOLD,
        DEFAULT_LLM_THRESHOLD,
    )
    unreviewed_count = len(dedupe_image_paths) - len(reviewed_paths)
    top_tags = Counter(dict(tag_counts.most_common(20)))
    large_image_paths = [
        path for path in image_paths if path.stat().st_size > large_image_threshold
    ]
    largest_paths = {
        *sorted(
            large_image_paths,
            key=lambda path: (-image_dimensions(path)[0], path),
        )[:5],
        *sorted(
            large_image_paths,
            key=lambda path: (-image_dimensions(path)[1], path),
        )[:5],
        *sorted(
            large_image_paths,
            key=lambda path: (-path.stat().st_size, path),
        )[:5],
    }

    sections = [
        [f"Number of images: {len(image_paths)}"],
        _format_report_counts("Extensions", extension_counts),
        _format_report_counts("Categories", category_counts),
        _format_report_counts("Genres", genre_counts),
        _format_report_counts("Tags", top_tags),
    ]
    outstanding_checks: list[str] = []
    if missing_metadata_count:
        outstanding_checks.append(
            f"Number of images not in metadata: {missing_metadata_count}"
        )
    if unreviewed_count:
        outstanding_checks.append(
            f"Number of images not reviewed for dupes: {unreviewed_count}"
        )
    if clean_filename_mismatches:
        outstanding_checks.append(
            "Number of images where the filename does not match the clean filename: "
            f"{len(clean_filename_mismatches)}"
        )
        outstanding_checks.extend(
            _format_filename_mismatches(
                directory_path,
                clean_filename_mismatches,
            )
        )
    if gallery_is_out_of_date(directory_path, image_paths):
        outstanding_checks.append(f"{GALLERY_NAME.name} is out of date.")
    if outstanding_checks:
        sections.append(outstanding_checks)
    if largest_paths:
        sections.append(
            [
                "Largest Images:",
                "",
                *_format_largest_images(directory_path, largest_paths),
            ]
        )
    return "\n\n".join("\n".join(section) for section in sections if section)
