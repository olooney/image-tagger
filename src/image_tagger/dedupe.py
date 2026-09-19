import fnmatch
import os
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Literal

import jinja2
from PIL import Image
from send2trash import send2trash
from send2trash.exceptions import TrashPermissionError

from .compare import (
    ClipImageComparisonMethod,
    ImageDuplicateMatch,
    _mark_images_deduped,
    _reviewed_cached_image_paths,
    base64_encode_image,
    dedupe_new_image_matches,
)
from .constants import (
    CLIP_MODEL,
    DEDUPE_EMBEDDINGS_FILENAME,
    DEDUPE_REVIEW_FILENAME,
    DEFAULT_AUTOMATIC_THRESHOLD,
    DEFAULT_LLM_THRESHOLD,
)
from .util import Pathish, display_file_operation, quote_display_path
from .vision import VisionModelProvider


@dataclass(frozen=True)
class DedupeReviewEntry:
    """Describe one duplicate decision in the static review report."""

    left_path: Path
    right_path: Path
    left_image_src: str | None
    right_image_src: str | None
    score: float
    decision_source: str
    judgement_text: str | None
    action: str
    duplicate_side: Literal["left", "right"]
    is_removal: bool = False


def _dedupe_review_entry(
    match: ImageDuplicateMatch,
    *,
    action: str,
) -> DedupeReviewEntry:
    """Capture report thumbnails before a duplicate can be removed."""
    left_path = match.presented_left_path or match.left_path
    right_path = match.presented_right_path or match.right_path
    duplicate_side: Literal["left", "right"] = (
        "left" if match.right_path == left_path else "right"
    )
    return DedupeReviewEntry(
        left_path=left_path,
        right_path=right_path,
        left_image_src=_thumbnail_data_url(left_path),
        right_image_src=_thumbnail_data_url(right_path),
        score=match.score,
        decision_source=match.decision_source,
        judgement_text=match.judgement_text,
        action=action,
        duplicate_side=duplicate_side,
    )


def _duplicate_removal_action(
    duplicate_side: Literal["left", "right"],
    *,
    dry_run: bool,
    removed: bool = True,
) -> str:
    """Describe the review side selected for duplicate removal."""
    side = duplicate_side.capitalize()
    if not removed:
        return f"{side} not removed"
    if dry_run:
        return f"{side} would remove"
    return f"{side} removed"


def _thumbnail_data_url(image_path: Pathish) -> str | None:
    """Encode an image as a PNG data URL within a 500-by-750-pixel bound."""
    try:
        with Image.open(image_path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((500, 750), Image.Resampling.LANCZOS)
            return f"data:image/png;base64,{base64_encode_image(thumbnail)}"
    except (OSError, ValueError):
        return None


def generate_dedupe_review(
    entries: Sequence[DedupeReviewEntry],
    output_filename: Pathish,
) -> Path:
    """Render a static HTML report for duplicate-removal decisions."""
    template_text = (
        resources.files("image_tagger.data")
        .joinpath("dedupe_review.html")
        .read_text(encoding="utf-8")
    )
    template = jinja2.Environment(autoescape=True).from_string(template_text)
    output_path = Path(output_filename)
    sorted_entries = sorted(entries, key=lambda entry: not entry.is_removal)
    output_path.write_text(template.render(entries=sorted_entries), encoding="utf-8")
    return output_path


def dedupe_images(
    directory: Pathish,
    *,
    find_images_func: Callable[..., list[Path]] | None = None,
    dedupe_new_image_matches_func: Callable[..., list[ImageDuplicateMatch]]
    | None = None,
    clip_image_comparison_method_factory: Callable[..., ClipImageComparisonMethod]
    | None = None,
    send2trash_func: Callable[[Path], None] | None = None,
    automatic_threshold: float = DEFAULT_AUTOMATIC_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    filename_glob: str | None = None,
    verbose: int = 1,
    dry_run: bool = False,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    batch_size: int = 32,
) -> list[ImageDuplicateMatch]:
    """Send duplicate images from a directory to the recycle bin."""
    if find_images_func is None:
        from .tagging import find_images

        find_images_func = find_images
    if dedupe_new_image_matches_func is None:
        dedupe_new_image_matches_func = dedupe_new_image_matches
    if clip_image_comparison_method_factory is None:
        clip_image_comparison_method_factory = ClipImageComparisonMethod
    if send2trash_func is None:
        send2trash_func = send2trash

    directory_path = Path(directory)
    image_paths = sorted(find_images_func(directory_path))
    if verbose == 1:
        print(f"working in {quote_display_path(directory_path)}")
    cache_filename = directory_path / DEDUPE_EMBEDDINGS_FILENAME
    if filename_glob is None:
        processed_image_paths = _reviewed_cached_image_paths(
            image_paths,
            cache_filename,
            CLIP_MODEL,
            automatic_threshold,
            llm_threshold,
        )
        new_image_paths = [
            path for path in image_paths if path not in processed_image_paths
        ]
    else:
        new_image_paths = [
            path for path in image_paths if fnmatch.fnmatch(path.name, filename_glob)
        ]
        processed_image_paths = [
            path for path in image_paths if path not in new_image_paths
        ]
    rejected_llm_matches: list[ImageDuplicateMatch] = []
    comparison_method = (
        clip_image_comparison_method_factory(
            cache_filename=cache_filename,
        )
        if new_image_paths
        else None
    )
    matches = dedupe_new_image_matches_func(
        new_image_paths,
        processed_image_paths,
        automatic_threshold=automatic_threshold,
        llm_threshold=llm_threshold,
        provider=provider,
        batch_size=batch_size,
        verbose=verbose,
        rejected_llm_matches=rejected_llm_matches,
        comparison_method=comparison_method,
    )

    removed_paths: set[Path] = set()
    removal_failed = False
    review_entries = [
        _dedupe_review_entry(
            match,
            action="Kept both",
        )
        for match in rejected_llm_matches
    ]
    for match in matches:
        duplicate = match.right_path
        kept = match.left_path
        if duplicate in removed_paths or kept in removed_paths:
            continue
        review_entry = _dedupe_review_entry(match, action="")
        action = _duplicate_removal_action(
            review_entry.duplicate_side,
            dry_run=dry_run,
        )
        removed = False
        if verbose >= 1:
            print(
                display_file_operation(
                    "removing duplicate",
                    duplicate,
                    kept,
                    verbose=verbose,
                    relative_to=directory_path,
                ),
                end="",
            )
        try:
            if not dry_run:
                try:
                    send2trash_func(duplicate)
                except TrashPermissionError:
                    duplicate.unlink()
            removed_paths.add(duplicate)
            removed = not dry_run
            if verbose >= 1:
                print("success!")
        except Exception:
            removal_failed = True
            action = _duplicate_removal_action(
                review_entry.duplicate_side,
                dry_run=False,
                removed=False,
            )
            if verbose >= 1:
                print("error!")
            else:
                print(f"error removing {os.fspath(duplicate)!r}!")
            traceback.print_exc()
        review_entries.append(
            replace(
                review_entry,
                action=action,
                is_removal=removed,
            )
        )

    if not dry_run and not removal_failed and filename_glob is None:
        _mark_images_deduped(
            (path for path in image_paths if path.exists()),
            cache_filename,
            CLIP_MODEL,
            automatic_threshold,
            llm_threshold,
        )

    review_filename = directory_path / DEDUPE_REVIEW_FILENAME
    if review_entries:
        generate_dedupe_review(review_entries, review_filename)
        if verbose >= 1:
            print(f"wrote {quote_display_path(review_filename)}")
    else:
        review_filename.unlink(missing_ok=True)

    return matches
