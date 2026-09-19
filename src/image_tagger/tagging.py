import csv
import json
import os
import re
import time
import traceback
import uuid
from collections.abc import Callable, Hashable, Iterable, Mapping
from datetime import datetime
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import pandas as pd
import requests
from PIL import Image
from pydantic import BaseModel, create_model

from .compare import base64_encode_image, resize_image_to_fit
from .constants import CSV_COLUMNS, REVIEW_ID_COLUMN, WELCOME_EXTENSIONS
from .util import Pathish
from .vision import (
    VisionModelClientAdapter,
    VisionModelProvider,
    get_vision_model_client_adapter,
)


class ImageTagData(BaseModel):
    """Structured metadata returned by vision models."""

    description: str
    category: str
    genre: str
    tags: list[str]
    filename_already_makes_sense: bool
    filename: str


IMAGE_PROMPT_TEMPLATE: str = (
    resources.files("image_tagger.data").joinpath("image_prompt.md").read_text()
)


# Backward-compatible alias used throughout tests and CLI code.
csv_columns = CSV_COLUMNS


def image_tag_response_model(categories: list[str]) -> type[BaseModel]:
    """Build the tagging schema for the configured shelf aliases."""
    if not categories:
        raise ValueError(
            "stack map must define at least one non-default shelf for tagging."
        )
    category_type = cast(Any, Literal)[tuple(categories)]
    return create_model(
        "ConfiguredImageTagData",
        __base__=ImageTagData,
        category=(category_type, ...),
    )


def _format_categories(
    categories: list[str],
    descriptions: Mapping[str, str],
) -> str:
    """Format configured category identifiers and optional guidance."""
    return ", ".join(
        f'"{category}": {descriptions[category]}'
        if category in descriptions
        else f'"{category}"'
        for category in categories
    )


def new_metadata_review_id() -> str:
    """Return a time-ordered stable metadata row ID."""
    return str(uuid.uuid7())


def ensure_metadata_review_ids(metadata_filename: Pathish) -> bool:
    """Persist unique UUID7 IDs for metadata rows when migration is needed."""
    metadata_path = Path(metadata_filename)
    if not metadata_path.is_file():
        return False
    try:
        metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        metadata_df = pd.DataFrame(columns=csv_columns)

    changed = REVIEW_ID_COLUMN not in metadata_df
    if changed:
        metadata_df[REVIEW_ID_COLUMN] = ""
    used_ids: set[str] = set()
    for index, value in metadata_df[REVIEW_ID_COLUMN].items():
        review_id = str(value).strip()
        if not review_id or review_id in used_ids:
            review_id = new_metadata_review_id()
            metadata_df.at[index, REVIEW_ID_COLUMN] = review_id
            changed = True
        used_ids.add(review_id)
    if changed:
        metadata_df.to_csv(metadata_path, index=False)
    return changed


def ensure_metadata_columns(
    metadata_filename: Pathish,
    columns: Iterable[str],
) -> list[str]:
    """Add missing metadata columns and return the resulting header."""
    metadata_path = Path(metadata_filename)
    if not metadata_path.is_file():
        return [
            *csv_columns,
            *[column for column in columns if column not in csv_columns],
        ]
    try:
        metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        metadata_df = pd.DataFrame(columns=csv_columns)
    changed = False
    for column in columns:
        if column not in metadata_df:
            metadata_df[column] = ""
            changed = True
    if changed:
        metadata_df.to_csv(metadata_path, index=False)
    return list(metadata_df.columns)


def clean_filename(filename: str) -> str:
    """Clean up a suggested filename."""
    filename = filename.lower()
    filename = re.sub(r"^[^a-zA-Z_]+", "", filename)
    filename = re.sub(r"[\s_-]+", "_", filename)
    filename = re.sub(r"[^a-zA-Z0-9_.]", "", filename)
    filename = re.sub(r"[\s_-]*\.+", ".", filename)

    return filename


def fix_extension(current_filename: str, suggested_filename: str) -> str:
    """Force a suggested filename to keep its original extension."""
    current_path = Path(current_filename)
    suggested_path = Path(suggested_filename)
    if current_path.suffix.lower() != suggested_path.suffix.lower():
        suggested_path = suggested_path.with_suffix(current_path.suffix)
    return suggested_path.name


def update_metadata_filepaths(
    metadata_filename: Pathish,
    file_changes: Iterable[tuple[Path, Path]],
) -> None:
    """Update metadata rows after image filenames change."""
    metadata_path = Path(metadata_filename)
    ensure_metadata_review_ids(metadata_path)
    if not metadata_path.is_file():
        return

    final_paths: dict[Path, Path] = {}
    for source, target in file_changes:
        for original_path, current_path in final_paths.items():
            if current_path == source:
                final_paths[original_path] = target
        final_paths[source] = target

    if not final_paths:
        return

    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    metadata_updated = False
    for index, row in metadata_df.iterrows():
        source = Path(row["original_filepath"])
        target = final_paths.get(source)
        if target is None:
            continue

        metadata_df.at[index, "original_filepath"] = os.fspath(target)
        metadata_df.at[index, "original_filename"] = target.name
        cleaned = str(row["clean_filename"])
        if cleaned:
            metadata_df.at[index, "clean_filename"] = fix_extension(
                target.name,
                cleaned,
            )
        metadata_updated = True

    if metadata_updated:
        metadata_df.to_csv(metadata_path, index=False)


def tag_image(
    filepath: Pathish,
    client_adapter: VisionModelClientAdapter,
    prompt_template: str = IMAGE_PROMPT_TEMPLATE,
    categories: list[str] | None = None,
    category_descriptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Tag a single image with a vision model."""
    filepath_string = os.fspath(filepath)
    if filepath_string.startswith("http"):
        url = filepath_string
        filename = urlsplit(url).path.split("/")[-1]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = resize_image_to_fit(image)
    else:
        image_path = Path(filepath)
        filename = image_path.name
        image = resize_image_to_fit(image_path)

    base64_image_data = base64_encode_image(image)

    prompt = prompt_template.format(
        filename=filename,
        categories=_format_categories(categories or [], category_descriptions or {}),
    )
    vision_start_time = time.perf_counter()
    response_format = (
        image_tag_response_model(categories) if categories is not None else ImageTagData
    )
    response = client_adapter.vision_task(base64_image_data, prompt, response_format)
    vision_duration = time.perf_counter() - vision_start_time
    data = response.data.model_dump()

    suggested_filename_value = data["filename"]
    if not isinstance(suggested_filename_value, str):
        raise TypeError("Vision response filename must be a string.")
    suggested_filename = clean_filename(suggested_filename_value)
    suggested_filename_fixed = fix_extension(filename, suggested_filename)

    data["clean_filename"] = suggested_filename_fixed
    data["original_filepath"] = filepath_string
    data["original_filename"] = filename
    data["total_tokens"] = response.total_tokens
    data["provider_name"] = client_adapter.provider_name
    data["model"] = response.model
    data["width"] = image.size[0]
    data["height"] = image.size[1]
    data["vision_duration"] = vision_duration

    return data


def tag_images(
    filepaths: Iterable[Pathish],
    output_filename: Pathish,
    retry_errors: bool = False,
    verbose: int = 1,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    instructions_filename: Pathish | None = None,
    categories: list[str] | None = None,
    category_descriptions: Mapping[str, str] | None = None,
    quad_detector: Callable[[Path, VisionModelClientAdapter, int], list[list[float]]]
    | None = None,
) -> None:
    """Tag images and write metadata rows."""
    output_path = Path(output_filename)
    ensure_metadata_review_ids(output_path)
    client_adapter = get_vision_model_client_adapter(provider)
    if instructions_filename is None:
        prompt_template = IMAGE_PROMPT_TEMPLATE
    else:
        prompt_template = Path(instructions_filename).read_text(encoding="utf-8")
    if verbose >= 1:
        print(f"Using {client_adapter}")
    file_already_exists = output_path.exists()
    mode = "a" if file_already_exists else "w"
    columns = (
        ensure_metadata_columns(output_path, ["quad"])
        if quad_detector is not None
        else csv_columns
    )

    try:
        with output_path.open(mode, newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            if not file_already_exists:
                writer.writeheader()

            vision_durations = []

            for index, filepath in enumerate(filepaths):
                row_start_time = time.perf_counter()

                try:
                    row = tag_image(
                        filepath,
                        client_adapter,
                        prompt_template,
                        categories,
                        category_descriptions,
                    )
                    duration = row.pop("vision_duration")
                    vision_durations.append(duration)
                    row["tags"] = ";".join(tag.lower().strip() for tag in row["tags"])
                    if quad_detector is not None:
                        row["quad"] = json.dumps(
                            quad_detector(Path(filepath), client_adapter, verbose)
                        )
                    row.update(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "status": "ok",
                            REVIEW_ID_COLUMN: new_metadata_review_id(),
                        }
                    )
                    writer.writerow(row)
                    csv_file.flush()

                    if verbose == 0:
                        print(".", end=("\n" if (index + 1) % 100 == 0 else ""))
                    elif verbose == 1:
                        average_durations = (
                            vision_durations[1:]
                            if len(vision_durations) > 2
                            else vision_durations
                        )
                        average_duration = sum(average_durations) / len(
                            average_durations
                        )
                        print(
                            f"{row['timestamp']} {row['original_filename']} -> "
                            f"{row['clean_filename']}: {row['category']} {row['genre']} {row['status']} "
                            f"{duration:0.2f}s avg {average_duration:0.2f}s"
                        )
                    elif verbose >= 2:
                        print(repr(row))
                except KeyboardInterrupt:
                    if verbose >= 1:
                        print("\nInterrupted; cleaning up...")
                    raise
                except Exception:
                    error_message = traceback.format_exc()
                    duration = time.perf_counter() - row_start_time

                    if verbose == 1:
                        print("e", end=("\n" if (index + 1) % 100 == 0 else ""))
                    elif verbose == 2:
                        original_filename = Path(filepath).name
                        print(
                            f"{datetime.now().isoformat()} {original_filename} -> "
                            f"<none> error {duration:0.2f}s"
                        )
                    elif verbose >= 3:
                        print(error_message)

                    writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "original_filepath": filepath,
                            "status": "error",
                            "description": error_message,
                            REVIEW_ID_COLUMN: new_metadata_review_id(),
                        }
                    )
    finally:
        client_adapter.cleanup()


def populate_metadata_quads(
    metadata_filename: Pathish,
    *,
    quad_detector: Callable[[Path, VisionModelClientAdapter, int], list[list[float]]],
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    verbose: int = 1,
) -> int:
    """Populate blank quad values for tagged metadata rows with existing images."""
    metadata_path = Path(metadata_filename)
    if not metadata_path.is_file():
        return 0
    ensure_metadata_columns(metadata_path, ["quad"])
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    rows_to_process: list[tuple[Hashable, Path]] = []
    for index, raw_row in metadata_df.iterrows():
        row = cast("pd.Series[Any]", raw_row)
        if str(row.get("status", "")).strip() != "ok" or str(row["quad"]).strip():
            continue
        original_filepath = str(row.get("original_filepath", "")).strip()
        if not original_filepath:
            continue
        original_path = Path(original_filepath)
        clean_filename = str(row.get("clean_filename", "")).strip()
        candidate_paths = [original_path]
        if clean_filename:
            candidate_paths.append(original_path.with_name(clean_filename))
        image_path = next((path for path in candidate_paths if path.is_file()), None)
        if image_path is not None:
            rows_to_process.append((index, image_path))

    if not rows_to_process:
        return 0
    client_adapter = get_vision_model_client_adapter(provider)
    if verbose >= 1:
        print(f"Using {client_adapter}")
    completed_count = 0
    try:
        for row_number, (index, image_path) in enumerate(rows_to_process):
            try:
                metadata_df.at[index, "quad"] = json.dumps(
                    quad_detector(image_path, client_adapter, verbose)
                )
                completed_count += 1
                if verbose == 0:
                    print(".", end="\n" if (row_number + 1) % 100 == 0 else "")
            except KeyboardInterrupt:
                raise
            except Exception:
                if verbose == 0:
                    print("e", end="\n" if (row_number + 1) % 100 == 0 else "")
                elif verbose >= 2:
                    traceback.print_exc()
        if verbose == 0:
            print()
    finally:
        client_adapter.cleanup()
    if completed_count:
        metadata_df.to_csv(metadata_path, index=False)
    return completed_count


def previously_tagged_filenames(metadata_filename: Pathish) -> set[str]:
    """Return filenames already tagged successfully."""
    metadata_path = Path(metadata_filename)
    if not metadata_path.exists():
        return set()

    tagged_filenames: set[str] = set()
    with metadata_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("status") != "ok":
                continue

            for column in ["original_filename", "clean_filename"]:
                value = row.get(column)
                if not value:
                    continue
                tagged_filenames.add(Path(value).name)

    return tagged_filenames


def find_images(
    dirs: Pathish | Iterable[Pathish],
    max_days_old: float | None = None,
    metadata_filename: Pathish | None = None,
    extension_filter: Iterable[str] | None = WELCOME_EXTENSIONS,
) -> list[Path]:
    """Find untagged image files in directories recursively."""
    if max_days_old is None:
        max_days_old = float("Inf")

    if isinstance(dirs, (str, os.PathLike)):
        directories = [dirs]
    else:
        directories = dirs

    tagged_filenames = (
        previously_tagged_filenames(metadata_filename)
        if metadata_filename is not None
        else set()
    )
    allowed_extensions = (
        {extension.lower() for extension in extension_filter}
        if extension_filter is not None
        else None
    )
    current_time = time.time()

    filepaths: list[Path] = []
    for directory in directories:
        directory_path = Path(cast("Any", directory))
        for filepath in directory_path.rglob("*"):
            if not filepath.is_file():
                continue
            if (current_time - filepath.stat().st_mtime) >= max_days_old * 86400:
                continue
            if allowed_extensions is not None:
                if filepath.suffix.lower() not in allowed_extensions:
                    continue
            if filepath.name in tagged_filenames:
                continue
            filepaths.append(filepath)

    return filepaths
