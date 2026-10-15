import shutil
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .constants import IMAGE_EXTENSIONS
from .util import Pathish, display_file_operation
from .wall import image_aspect_ratios, infer_wall_layout

register_heif_opener()

DEFAULT_MAX_WIDTH: int = 400
DEFAULT_MAX_HEIGHT: int = 600
PUBLISH_FILE_EXTENSIONS: set[str] = {".csv", ".npz"}
PUBLISH_FILENAMES: set[str] = {"index.html"}


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


def _format_summary(
    success_count: int,
    metadata_count: int,
    error_count: int,
    skip_count: int,
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
    return ", ".join(counts)


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
    finally:
        summary = _format_summary(
            success_count,
            metadata_count,
            error_count,
            skip_count,
        )
        if verbose >= 1 and summary:
            if verbose == 1 and success_count + error_count:
                print()
            print(summary)