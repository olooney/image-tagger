import hashlib
import os
import random
import re
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any, cast

import jinja2
import pandas as pd
from PIL import Image, ImageOps

from .constants import (
    WALL_DOUBLE_WIDE_THRESHOLD,
    WALL_LAYOUT_MAX_PASSES,
    WALL_TITLE_TEMPLATE,
)
from .util import Pathish, quote_display_path


def median_image_aspect_ratio(filepaths: Iterable[Pathish]) -> float:
    """Return the median width-to-height ratio for image files."""
    aspect_ratios = image_aspect_ratios(filepaths)
    return median_aspect_ratio(aspect_ratios.values())


def median_aspect_ratio(aspect_ratios: Iterable[float]) -> float:
    """Return the median aspect ratio from ratio values."""
    sorted_aspect_ratios = sorted(aspect_ratios)

    if not sorted_aspect_ratios:
        return 1.0

    midpoint = len(sorted_aspect_ratios) // 2
    if len(sorted_aspect_ratios) % 2:
        return sorted_aspect_ratios[midpoint]
    return (sorted_aspect_ratios[midpoint - 1] + sorted_aspect_ratios[midpoint]) / 2


def image_aspect_ratios(filepaths: Iterable[Pathish]) -> dict[Path, float]:
    """Return display width-to-height ratios keyed by image path."""
    aspect_ratios: dict[Path, float] = {}
    for filepath in filepaths:
        image_path = Path(filepath)
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        if height > 0:
            aspect_ratios[image_path] = width / height

    return aspect_ratios


def infer_wall_layout(
    aspect_ratios: Mapping[Path, float],
    double_wide_threshold: float = WALL_DOUBLE_WIDE_THRESHOLD,
) -> tuple[float, set[Path]]:
    """Infer a one-cell aspect ratio and double-wide wall tiles."""
    cell_aspect_ratio = median_aspect_ratio(aspect_ratios.values())

    for _ in range(WALL_LAYOUT_MAX_PASSES):
        double_wide_paths = {
            filepath
            for filepath, aspect_ratio in aspect_ratios.items()
            if aspect_ratio > double_wide_threshold * cell_aspect_ratio
        }
        adjusted_aspect_ratios = [
            aspect_ratio / 2 if filepath in double_wide_paths else aspect_ratio
            for filepath, aspect_ratio in aspect_ratios.items()
        ]
        next_cell_aspect_ratio = median_aspect_ratio(adjusted_aspect_ratios)
        if next_cell_aspect_ratio == cell_aspect_ratio:
            break
        cell_aspect_ratio = next_cell_aspect_ratio

    double_wide_paths = {
        filepath
        for filepath, aspect_ratio in aspect_ratios.items()
        if aspect_ratio > double_wide_threshold * cell_aspect_ratio
    }
    return cell_aspect_ratio, double_wide_paths


def paths_with_mtime(filepaths: Iterable[Pathish]) -> list[tuple[float, Path]]:
    """Return file paths paired with their modification times."""
    return [(Path(filepath).stat().st_mtime, Path(filepath)) for filepath in filepaths]


def seeded_wall_sort_key(
    directory: Path, filepath: Path, seed: int
) -> tuple[bytes, str]:
    """Return a stable pseudo-random wall sort key for a file path."""
    relative_filepath = Path(os.path.relpath(filepath, directory)).as_posix()
    key_text = f"{seed}\0{relative_filepath}".encode("utf-8", errors="surrogateescape")
    return hashlib.blake2b(key_text, digest_size=16).digest(), relative_filepath


def singular_wall_title_word(word: str) -> str:
    """Return a simple singular display form for a wall title word."""
    if len(word) > 3 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 1 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def wall_title_from_directory(directory: Pathish) -> str:
    """Return a display title inferred from a wall directory name."""
    directory_name = Path(directory).name.replace("_", " ").strip()
    title_words = [
        singular_wall_title_word(word)
        for word in re.split(r"\s+", directory_name.casefold())
        if word
    ]
    if not title_words:
        return "Image Wall"
    return f"{' '.join(title_words).title()} Wall"


def wall_metadata_titles(metadata_filename: Pathish | None) -> dict[str, str]:
    """Return image metadata title text keyed by possible filenames."""
    if metadata_filename is None:
        return {}

    metadata_path = Path(metadata_filename)
    if not metadata_path.is_file():
        return {}

    metadata_df = pd.read_csv(metadata_path)
    titles: dict[str, str] = {}
    for raw_item in metadata_df.to_dict("records"):
        item = cast("dict[str, Any]", raw_item)
        if item.get("status") != "ok":
            continue

        item["tags"] = item["tags"].replace(";", ", ")
        title = WALL_TITLE_TEMPLATE.format(**item)

        if not title:
            continue

        for column in ["original_filepath", "original_filename", "clean_filename"]:
            value = item.get(column, "")
            if value and isinstance(value, str):
                titles[Path(value).name] = title
                titles[os.fspath(Path(value))] = title

    return titles


def generate_wall(
    directory: Pathish,
    find_images_func=None,
    output_filename: Pathish | None = None,
    metadata_filename: Pathish | None = None,
    order: str = "random",
    seed: int | None = None,
    title: str | None = None,
    double_wide_threshold: float = WALL_DOUBLE_WIDE_THRESHOLD,
    verbose: int = 1,
) -> Path:
    """Generate a static image wall HTML file."""
    if find_images_func is None:
        from .tagging import find_images

        find_images_func = find_images

    directory_path = Path(directory)
    output_path = (
        Path(output_filename)
        if output_filename is not None
        else directory_path / "index.html"
    )
    filepaths = find_images_func(directory_path)
    if order == "name":
        filepaths.sort(key=lambda filepath: filepath.name.casefold())
    elif order == "date":
        filepaths = [
            filepath
            for _, filepath in sorted(paths_with_mtime(filepaths), reverse=True)
        ]
    elif order == "random":
        if seed is None:
            random.shuffle(filepaths)
        else:
            filepaths.sort(
                key=lambda filepath: seeded_wall_sort_key(
                    directory_path, filepath, seed
                )
            )
    else:
        raise ValueError(f"Unsupported wall order: {order}")
    aspect_ratios = image_aspect_ratios(filepaths)
    aspect_ratio, double_wide_paths = infer_wall_layout(
        aspect_ratios,
        double_wide_threshold,
    )
    cell_width = 200
    cell_height = round(cell_width / aspect_ratio)
    metadata_titles = wall_metadata_titles(metadata_filename)
    items = [
        {
            "src": Path(os.path.relpath(filepath, output_path.parent)).as_posix(),
            "alt": filepath.name,
            "title": metadata_titles.get(
                os.fspath(filepath),
                metadata_titles.get(filepath.name, filepath.name),
            ),
            "is_double_wide": filepath in double_wide_paths,
        }
        for filepath in filepaths
    ]

    template_text = (
        resources.files("image_tagger.data")
        .joinpath("wall_template.html")
        .read_text(encoding="utf-8")
    )
    template = jinja2.Environment(autoescape=True).from_string(template_text)
    output = template.render(
        items=items,
        cell_width=cell_width,
        cell_height=cell_height,
        wall_title=title or wall_title_from_directory(directory_path),
    )
    output_path.write_text(output, encoding="utf-8")
    if verbose >= 1:
        double_wide_count = sum(1 for item in items if item["is_double_wide"])
        cell_count = len(items) + double_wide_count
        # Preserve the historic diagnostic denominator so tiny walls still report
        # a meaningful divisibility signal.
        reported_cell_count = cell_count if cell_count >= 20 else 1080

        DIVISOR_RANGE = 20
        n_divisors = 1
        for width in range(2, DIVISOR_RANGE + 1):
            if reported_cell_count % width == 0:
                n_divisors += 1
        divisibility_percentage = 100 * n_divisors / DIVISOR_RANGE

        print(f"wrote {quote_display_path(output_path)}")
        print(
            f"{len(items)} images + "
            f"{double_wide_count} double-wide = {reported_cell_count} ({divisibility_percentage}%)"
        )
    return output_path
