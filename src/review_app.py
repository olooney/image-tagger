import fnmatch
import html
import json
import logging
import os
import re
import shutil
import socket
import time
import webbrowser
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote

import jinja2
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from PIL import Image
from pydantic import BaseModel, Field
from send2trash import send2trash

import image_tagger as it
import transform
from constants import WELCOME_EXTENSIONS
from stackmap import StackMap
from util import Pathish, make_unique


template_environment = jinja2.Environment(
    loader=jinja2.PackageLoader("image_tagger_data", package_path=""),
    autoescape=True,
)
page_template = template_environment.get_template("review.html")
card_template = template_environment.get_template("review_card.html")
app = FastAPI(title="Image Metadata Review")
LOGGER: logging.Logger = logging.getLogger(__name__)
REVIEW_ID_COLUMN: str = it.REVIEW_ID_COLUMN
FULL_QUAD: str = json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]])
GENRE_OPTIONS: list[str] = [
    "sci-fi",
    "fantasy",
    "comedy",
    "mystery",
    "horror",
    "drama",
    "tragedy",
    "nonfiction",
    "nature",
    "abstract",
]


class ClipRequest(BaseModel):
    """Interactive crop and optional resize parameters."""

    points: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    background: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    mode: Literal["perspective", "rectangle"] = "perspective"
    output_width: int | None = Field(default=None, ge=1)
    output_height: int | None = Field(default=None, ge=1)
    resampling: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "lanczos"


def set_review_metadata(
    metadata_filename: Pathish,
    stackmap: StackMap,
    provider: it.VisionModelProvider | str = it.VisionModelProvider.OPENAI,
    verbose: int = 1,
    filename_glob: str | None = None,
) -> None:
    """Set the metadata file used by the review app."""
    app.state.metadata_path = Path(metadata_filename)
    app.state.stackmap = stackmap
    app.state.provider = it.VisionModelProvider(provider)
    app.state.verbose = verbose
    app.state.filename_glob = filename_glob


def review_metadata_path() -> Path:
    """Return the configured review metadata path."""
    metadata_path = getattr(app.state, "metadata_path", None)
    if metadata_path is None:
        raise RuntimeError("Review metadata path has not been configured.")
    return cast("Path", metadata_path)


def review_stackmap() -> StackMap:
    """Return the configured shelf map."""
    stackmap = getattr(app.state, "stackmap", None)
    if stackmap is None:
        raise RuntimeError("Review stack map has not been configured.")
    return cast("StackMap", stackmap)


def review_provider() -> it.VisionModelProvider:
    """Return the configured review vision-model provider."""
    return cast(
        "it.VisionModelProvider",
        getattr(app.state, "provider", it.VisionModelProvider.OPENAI),
    )


def review_verbose() -> int:
    """Return the configured review verbosity."""
    return cast("int", getattr(app.state, "verbose", 1))


def review_filename_glob() -> str | None:
    """Return the configured review filename glob."""
    return cast("str | None", getattr(app.state, "filename_glob", None))


def first_available_port(start_port: int = 8001) -> int:
    """Return the first available localhost TCP port."""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue
            return port


def review_category_options() -> list[str]:
    """Return category options from the configured shelf map."""
    return review_stackmap().categories


def review_genre_options(metadata_path: Path) -> list[str]:
    """Return genre options from the tagging prompt."""
    if not metadata_path.is_file():
        return []
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    genres = list(GENRE_OPTIONS)
    if "genre" not in metadata_df:
        return genres
    for genre in metadata_df["genre"]:
        genre_value = str(genre).strip()
        if genre_value and genre_value not in genres:
            genres.append(genre_value)
    return genres


def current_image_path(row: pd.Series[Any]) -> Path | None:
    """Return the existing image path for a metadata row."""
    original_path = Path(row["original_filepath"])
    clean_filename = str(row.get("clean_filename", "")).strip()
    clean_path = original_path.with_name(clean_filename) if clean_filename else None
    if original_path.is_file():
        return original_path
    if clean_path is not None and clean_path.is_file():
        return clean_path
    return None


def review_directory_images(directory: Path) -> list[Path]:
    """Return all reviewable images below a directory."""
    return sorted(
        it.find_images(
            directory,
            extension_filter=WELCOME_EXTENSIONS,
        )
    )


def ensure_review_metadata(metadata_path: Path, image_paths: list[Path]) -> None:
    """Ensure every reviewable image has an editable metadata row."""
    it.ensure_metadata_review_ids(metadata_path)
    if metadata_path.is_file():
        try:
            metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
        except pd.errors.EmptyDataError:
            metadata_df = pd.DataFrame(columns=it.csv_columns)
    else:
        metadata_df = pd.DataFrame(columns=it.csv_columns)

    for column in it.csv_columns:
        if column not in metadata_df:
            metadata_df[column] = ""

    images_by_name: dict[str, list[Path]] = {}
    for image_path in image_paths:
        images_by_name.setdefault(image_path.name.casefold(), []).append(image_path)

    represented_paths: set[Path] = set()
    for index, raw_row in metadata_df.iterrows():
        row = cast("pd.Series[Any]", raw_row)
        if row.get("status") != "ok":
            continue
        image_path = current_image_path(row)
        if image_path is None:
            candidate_names = [
                str(row.get(column, "")).strip()
                for column in ("clean_filename", "original_filename")
            ]
            candidates = {
                candidate.resolve()
                for filename in candidate_names
                for candidate in images_by_name.get(Path(filename).name.casefold(), [])
                if filename
            }
            if len(candidates) == 1:
                image_path = candidates.pop()
                metadata_df.at[index, "original_filepath"] = str(image_path)
        if image_path is not None:
            represented_paths.add(image_path.resolve())

    new_rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        if image_path.resolve() in represented_paths:
            continue
        row = dict.fromkeys(it.csv_columns, "")
        row.update(
            {
                REVIEW_ID_COLUMN: it.new_metadata_review_id(),
                "status": "ok",
                "original_filepath": str(image_path.resolve()),
                "original_filename": image_path.name,
            }
        )
        new_rows.append(row)

    if new_rows:
        metadata_df = pd.concat(
            [metadata_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)


def metadata_row_index(metadata_df: pd.DataFrame, row_id: str) -> int:
    """Resolve a stable review ID, with positional fallback for legacy metadata."""
    if REVIEW_ID_COLUMN in metadata_df:
        matches = metadata_df.index[metadata_df[REVIEW_ID_COLUMN].astype(str) == row_id]
        if len(matches) == 1:
            return int(matches[0])
        raise ValueError(f"Unknown row id: {row_id}")
    try:
        row_index = int(row_id) - 1
    except ValueError as error:
        raise ValueError(f"Unknown row id: {row_id}") from error
    if row_index < 0 or row_index >= len(metadata_df):
        raise ValueError(f"Unknown row id: {row_id}")
    return row_index


def review_row_image_path(metadata_path: Path, row_id: str) -> Path:
    """Return the current image path for one stable review row."""
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    row_index = metadata_row_index(metadata_df, row_id)
    image_path = current_image_path(cast("pd.Series[Any]", metadata_df.iloc[row_index]))
    if image_path is None:
        raise ValueError(f"Image not found for row id: {row_id}")
    return image_path


def detect_clip_corners(
    image_path: Path,
    algorithm: Literal["hough", "contour", "llm"] = "hough",
    provider: it.VisionModelProvider = it.VisionModelProvider.OPENAI,
    verbose: int = 1,
) -> tuple[tuple[int, int], list[list[float]], str | None]:
    """Return image dimensions and detected perspective corners."""
    with Image.open(image_path) as opened_image:
        source = opened_image.convert("RGB")
    background_hint = transform._background_hint(source)
    background: str | None = "#{:02x}{:02x}{:02x}".format(*background_hint.color_rgb)
    if algorithm == "llm":
        client = it.get_vision_model_client_adapter(provider)
        detection = transform.detect_crop(source, client=client, verbose=verbose)
        corners = detection.final_corners
        background_hint = detection.background
        background = "#{:02x}{:02x}{:02x}".format(*background_hint.color_rgb)
    else:
        image = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
        corners = (
            transform._detect_line_corners(image, background_hint)
            if algorithm == "hough"
            else transform._detect_contour_corners(image, background_hint)
        )
    if corners is None:
        corners = transform._full_corners(source.size)
    return source.size, corners.tolist(), background


def apply_clip(image_path: Path, request: ClipRequest) -> None:
    """Replace an image with an interactively selected perspective crop."""
    with Image.open(image_path) as opened_image:
        source = opened_image.convert("RGB")
    corners = np.asarray(request.points, dtype=np.float32)
    if not np.isfinite(corners).all():
        raise ValueError("The selected crop points do not form a valid quadrilateral.")
    corners = transform._order_corners(corners)
    if abs(cv2.contourArea(corners.reshape((-1, 1, 2)))) < 4:
        raise ValueError("The selected crop points do not form a valid quadrilateral.")
    if request.mode == "rectangle":
        left = max(0, int(np.floor(np.min(corners[:, 0]))))
        top = max(0, int(np.floor(np.min(corners[:, 1]))))
        right = min(source.width, int(np.ceil(np.max(corners[:, 0]))) + 1)
        bottom = min(source.height, int(np.ceil(np.max(corners[:, 1]))) + 1)
        if right - left < 2 or bottom - top < 2:
            raise ValueError("The selected rectangle is too small to crop.")
        clipped = source.crop((left, top, right, bottom))
        if request.output_width is not None and request.output_height is not None:
            resampling = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }[request.resampling]
            clipped = clipped.resize(
                (request.output_width, request.output_height),
                resample=resampling,
            )
    else:
        color = [int(request.background[index : index + 2], 16) for index in (1, 3, 5)]
        background = transform.BackgroundHint(
            color_rgb=color,
            confidence=1,
            uniformity="uniform",
        )
        clipped = transform._perspective_crop(source, corners, background=background)
        if clipped is None:
            raise ValueError("The selected crop could not be rendered.")

    temporary_path = image_path.with_name(
        f".{image_path.stem}.clip-{time.time_ns()}{image_path.suffix}"
    )
    try:
        clipped.save(temporary_path)
        send2trash(image_path)
        temporary_path.replace(image_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def review_items(metadata_path: Path) -> list[dict[str, Any]]:
    """Return editable review rows from a metadata CSV."""
    if not metadata_path.is_file():
        return []
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(metadata_df.to_dict("records"), start=1):
        item = cast("dict[str, Any]", raw_item)
        if item.get("status") != "ok":
            continue

        image_path = current_image_path(pd.Series(item))
        if image_path is None:
            continue
        filename_glob = review_filename_glob()
        if filename_glob is not None and not fnmatch.fnmatch(
            image_path.name,
            filename_glob,
        ):
            continue

        relative_path = os.path.relpath(image_path, metadata_path.parent)
        item["row_id"] = str(item.get(REVIEW_ID_COLUMN, "")).strip() or str(index)
        item["image_src"] = (
            f"/images/{quote(Path(relative_path).as_posix())}?v={time.time_ns()}"
        )
        item["current_filename"] = image_path.name
        item["clean_filename"] = str(item.get("clean_filename", ""))
        item["quad"] = str(item.get("quad", "")).strip()
        with Image.open(image_path) as source_image:
            item["display_width"], item["display_height"] = source_image.size
        item["tags_text"] = "\n".join(
            tag.strip()
            for tag in re.split(r"[;\r\n]+", str(item.get("tags", "")))
            if tag.strip()
        )
        notes = item.get("notes", "")
        item["notes"] = notes if notes and isinstance(notes, str) else ""
        items.append(item)
    return items


def rename_review_image(
    metadata_df: pd.DataFrame,
    row_index: int,
    requested_filename: str,
) -> str:
    """Rename the current image for a row and return the final filename."""
    row = cast("pd.Series[Any]", metadata_df.iloc[row_index])
    source = current_image_path(row)
    if source is None:
        return str(row.get("clean_filename", "")).strip()

    requested_filename = requested_filename.strip()
    if not requested_filename:
        return source.name

    target = source.with_name(requested_filename)
    if target == source:
        return source.name

    if source.suffix.lower() != target.suffix.lower():
        return source.name

    if target.exists():
        target = Path(make_unique(target))

    source.rename(target)
    return target.name


def write_metadata_row(metadata_path: Path, row_id: str, updates: dict[str, str]) -> None:
    """Write editable field updates for one stable review row."""
    it.ensure_metadata_review_ids(metadata_path)
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    if "quad" in updates:
        it.ensure_metadata_columns(metadata_path, ["quad"])
    editable_columns = {"category", "genre", "clean_filename", "tags", "description", "quad"}
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    row_index = metadata_row_index(metadata_df, row_id)

    if "clean_filename" in updates:
        updates["clean_filename"] = rename_review_image(
            metadata_df,
            row_index,
            updates["clean_filename"],
        )

    for column, value in updates.items():
        if column in editable_columns and column in metadata_df:
            if column == "tags":
                value = ";".join(
                    tag.strip()
                    for tag in re.split(r"[;\r\n]+", value)
                    if tag.strip()
                )
            metadata_df.at[row_index, column] = value.strip()

    backup_path = metadata_path.with_suffix(f"{metadata_path.suffix}.bak")
    shutil.copy2(metadata_path, backup_path)
    for attempt in range(5):
        try:
            metadata_df.to_csv(metadata_path, index=False)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.25)


def delete_metadata_row_image(metadata_path: Path, row_id: str) -> None:
    """Delete the current image for one stable review row and hide it from review."""
    it.ensure_metadata_review_ids(metadata_path)
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    row_index = metadata_row_index(metadata_df, row_id)

    row = cast("pd.Series[Any]", metadata_df.iloc[row_index])
    image_path = current_image_path(row)
    if image_path is not None:
        image_path.unlink()
    metadata_df.at[row_index, "status"] = "deleted"

    backup_path = metadata_path.with_suffix(f"{metadata_path.suffix}.bak")
    shutil.copy2(metadata_path, backup_path)
    for attempt in range(5):
        try:
            metadata_df.to_csv(metadata_path, index=False)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.25)


def render_card(
    item: dict[str, Any],
    categories: list[str],
    genres: list[str],
    saved: bool = False,
) -> str:
    """Render one review card."""
    return card_template.render(
        item=item,
        categories=categories,
        genres=genres,
        saved=saved,
    )


def render_review_list(
    items: list[dict[str, Any]],
    categories: list[str],
    genres: list[str],
) -> str:
    """Render the current review cards."""
    return "".join(render_card(item, categories, genres) for item in items)


def render_shelve_response(metadata_path: Path, output: str) -> str:
    """Render the shelve report and refreshed review card list."""
    if not metadata_path.is_file():
        categories = []
        genres = []
        items: list[dict[str, Any]] = []
    else:
        categories = review_category_options()
        genres = review_genre_options(metadata_path)
        items = review_items(metadata_path)
    report = f'<pre class="shelve-report">{html.escape(output)}</pre>'
    return "".join(
        [
            report,
            '<div id="review-list" hx-swap-oob="innerHTML">',
            render_review_list(items, categories, genres),
            "</div>",
        ]
    )


@app.get("/images/{image_path:path}")
async def image_file(image_path: str) -> FileResponse:
    """Serve an image relative to the metadata directory."""
    image_root = review_metadata_path().parent.resolve()
    resolved_path = (image_root / image_path).resolve()
    try:
        resolved_path.relative_to(image_root)
    except ValueError as error:
        raise HTTPException(status_code=404) from error
    if not resolved_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        resolved_path,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Render the metadata review page."""
    metadata_path = review_metadata_path()
    categories = review_category_options()
    genres = review_genre_options(metadata_path)
    items = review_items(metadata_path)
    return page_template.render(
        items=items,
        review_list=render_review_list(items, categories, genres),
        categories=categories,
        genres=genres,
        metadata_filename=metadata_path,
    )


@app.post("/row/{row_id}", response_class=HTMLResponse)
async def update_row(row_id: str, request: Request) -> str:
    """Update one one-based metadata row and return the refreshed card."""
    metadata_path = review_metadata_path()
    try:
        form = await request.form()
        write_metadata_row(
            metadata_path,
            row_id,
            {
                "category": str(form.get("category", "")),
                "genre": str(form.get("genre", "")),
                "clean_filename": str(form.get("clean_filename", "")),
                "tags": str(form.get("tags", "")),
                "description": str(form.get("description", "")),
            },
        )
        item = next(item for item in review_items(metadata_path) if item["row_id"] == row_id)
        return render_card(
            item,
            review_category_options(),
            review_genre_options(metadata_path),
            saved=True,
        )
    except Exception as error:
        LOGGER.exception("Failed to save review row %s.", row_id)
        return PlainTextResponse(f"Could not save: {error}", status_code=500)


@app.delete("/row/{row_id}", response_class=HTMLResponse)
async def delete_row(row_id: str) -> str:
    """Delete one reviewed image and return its final operation message."""
    metadata_path = review_metadata_path()
    image_path = review_row_image_path(metadata_path, row_id)
    delete_metadata_row_image(metadata_path, row_id)
    return (
        '<div class="alert alert-secondary delete-item-result" role="status">'
        f"Deleted {html.escape(image_path.name)}."
        "</div>"
    )


@app.get("/row/{row_id}/clip")
async def clip_details(
    row_id: str,
    algorithm: Literal["hough", "contour", "llm"] = "hough",
) -> dict[str, Any]:
    """Return source geometry and detected clip points without altering the image."""
    try:
        image_path = review_row_image_path(review_metadata_path(), row_id)
        (width, height), points, background = detect_clip_corners(
            image_path,
            algorithm,
            provider=review_provider(),
            verbose=review_verbose(),
        )
        return {
            "width": width,
            "height": height,
            "points": points,
            "background": background,
        }
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Failed to detect %s crop for row %s.", algorithm, row_id)
        detector_name = "VLM" if algorithm == "llm" else algorithm.upper()
        raise HTTPException(
            status_code=502,
            detail=f"{detector_name} detection failed: {error}",
        ) from error


@app.post("/row/{row_id}/clip")
async def clip_row(row_id: str, request: ClipRequest) -> dict[str, str | int]:
    """Apply an interactive perspective correction to one image."""
    try:
        metadata_path = review_metadata_path()
        image_path = review_row_image_path(metadata_path, row_id)
        apply_clip(image_path, request)
        write_metadata_row(metadata_path, row_id, {"quad": FULL_QUAD})
        relative_path = os.path.relpath(image_path, metadata_path.parent)
        with Image.open(image_path) as clipped_image:
            width, height = clipped_image.size
        return {
            "image_src": f"/images/{quote(Path(relative_path).as_posix())}",
            "width": width,
            "height": height,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/shelve", response_class=HTMLResponse)
async def shelve() -> str:
    """Shelve reviewed images and return the operation report."""
    metadata_path = review_metadata_path()
    stdout = StringIO()
    with redirect_stdout(stdout):
        it.shelve_images(metadata_path, stackmap=review_stackmap(), verbose=1)
    output = stdout.getvalue().strip() or "No images moved."
    return render_shelve_response(metadata_path, output)


@app.post("/row/{row_id}/shelve", response_class=HTMLResponse)
async def shelve_row(row_id: str) -> str:
    """Shelve one reviewed image and return its final operation message."""
    metadata_path = review_metadata_path()
    stdout = StringIO()
    with redirect_stdout(stdout):
        it.shelve_images(
            metadata_path,
            stackmap=review_stackmap(),
            verbose=2,
            review_ids={row_id},
            prune_verbose=0,
        )
    output = stdout.getvalue().strip() or "No image moved."
    return (
        '<div class="alert alert-success shelve-item-result" role="status">'
        f"{html.escape(output)}"
        "</div>"
    )


def review_metadata(
    metadata_filename: Pathish,
    *,
    stackmap: StackMap,
    provider: it.VisionModelProvider | str = it.VisionModelProvider.OPENAI,
    verbose: int = 1,
    filename_glob: str | None = None,
    start_port: int = 8001,
) -> None:
    """Serve a local metadata review app and open it in a browser."""
    import uvicorn

    metadata_path = Path(metadata_filename)
    it.ensure_metadata_review_ids(metadata_path)
    image_paths = review_directory_images(metadata_path.parent)
    if filename_glob is not None:
        image_paths = [
            image_path
            for image_path in image_paths
            if fnmatch.fnmatch(image_path.name, filename_glob)
        ]
    if not image_paths:
        print(f"No images found in {metadata_path.parent}.")
        return
    ensure_review_metadata(metadata_path, image_paths)
    backup_path = metadata_path.with_suffix(f"{metadata_path.suffix}.bak")
    set_review_metadata(
        metadata_path,
        stackmap,
        provider=provider,
        verbose=verbose,
        filename_glob=filename_glob,
    )
    port = first_available_port(start_port)
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    finally:
        if backup_path.exists():
            backup_path.unlink()
