import html
import logging
import os
import shutil
import socket
import time
import webbrowser
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import jinja2
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

import image_tagger as it
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
CATEGORY_OPTIONS: list[str] = [
    "ai",
    "art",
    "books",
    "comics",
    "diagrams",
    "horror",
    "hygge",
    "memes",
    "photography",
    "speculative",
    "vintage",
]
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


def set_review_metadata(metadata_filename: Pathish, stackmap: StackMap) -> None:
    """Set the metadata file used by the review app."""
    app.state.metadata_path = Path(metadata_filename)
    app.state.stackmap = stackmap


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


def review_category_options(metadata_path: Path) -> list[str]:
    """Return category options from the tagging prompt."""
    if not metadata_path.is_file():
        return []
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    categories = list(CATEGORY_OPTIONS)
    if "category" in metadata_df:
        for category in metadata_df["category"]:
            category_value = str(category).strip()
            if category_value and category_value not in categories:
                categories.append(category_value)
    return categories


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

        relative_path = os.path.relpath(image_path, metadata_path.parent)
        item["row_id"] = index
        item["image_src"] = f"/images/{quote(Path(relative_path).as_posix())}"
        item["current_filename"] = image_path.name
        item["clean_filename"] = str(item.get("clean_filename", "")) or image_path.name
        item["tags_text"] = str(item.get("tags", ""))
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


def write_metadata_row(metadata_path: Path, row_id: int, updates: dict[str, str]) -> None:
    """Write editable field updates for one one-based CSV row."""
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    editable_columns = {"category", "genre", "clean_filename", "tags", "description"}
    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    row_index = row_id - 1
    if row_index < 0 or row_index >= len(metadata_df):
        raise ValueError(f"Unknown row id: {row_id}")

    if "clean_filename" in updates:
        updates["clean_filename"] = rename_review_image(
            metadata_df,
            row_index,
            updates["clean_filename"],
        )

    for column, value in updates.items():
        if column in editable_columns and column in metadata_df:
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


def delete_metadata_row_image(metadata_path: Path, row_id: int) -> None:
    """Delete the current image for one one-based CSV row and hide it from review."""
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path, keep_default_na=False)
    row_index = row_id - 1
    if row_index < 0 or row_index >= len(metadata_df):
        raise ValueError(f"Unknown row id: {row_id}")

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
        categories = review_category_options(metadata_path)
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
    return FileResponse(resolved_path)


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Render the metadata review page."""
    metadata_path = review_metadata_path()
    categories = review_category_options(metadata_path)
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
async def update_row(row_id: int, request: Request) -> str:
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
            review_category_options(metadata_path),
            review_genre_options(metadata_path),
            saved=True,
        )
    except Exception as error:
        LOGGER.exception("Failed to save review row %s.", row_id)
        return PlainTextResponse(f"Could not save: {error}", status_code=500)


@app.delete("/row/{row_id}", response_class=HTMLResponse)
async def delete_row(row_id: int) -> str:
    """Delete one reviewed image and remove its card."""
    delete_metadata_row_image(review_metadata_path(), row_id)
    return ""


@app.post("/shelve", response_class=HTMLResponse)
async def shelve() -> str:
    """Shelve reviewed images and return the operation report."""
    metadata_path = review_metadata_path()
    stdout = StringIO()
    with redirect_stdout(stdout):
        it.shelve_images(metadata_path, stackmap=review_stackmap(), verbose=1)
    output = stdout.getvalue().strip() or "No images moved."
    return render_shelve_response(metadata_path, output)


def review_metadata(
    metadata_filename: Pathish,
    *,
    stackmap: StackMap,
    start_port: int = 8001,
) -> None:
    """Serve a local metadata review app and open it in a browser."""
    import uvicorn

    metadata_path = Path(metadata_filename)
    backup_path = metadata_path.with_suffix(f"{metadata_path.suffix}.bak")
    set_review_metadata(metadata_path, stackmap)
    port = first_available_port(start_port)
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    finally:
        if backup_path.exists():
            backup_path.unlink()
