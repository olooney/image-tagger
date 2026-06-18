import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

from PIL import Image
from pillow_heif import register_heif_opener

from constants import IMAGE_EXTENSIONS, UNWELCOME_EXTENSIONS, UPLOAD_DIR
from util import Pathish, display_file_operation, make_unique, quote_display_path

register_heif_opener()

EXTENSION_IMAGE_FORMATS: dict[str, set[str]] = {
    ".avif": {"AVIF"},
    ".bmp": {"BMP"},
    ".gif": {"GIF"},
    ".heic": {"HEIF"},
    ".jpeg": {"JPEG"},
    ".jpg": {"JPEG"},
    ".png": {"PNG"},
    ".tiff": {"TIFF"},
    ".webp": {"WEBP"},
}

IMAGE_FORMAT_EXTENSIONS: dict[str, str] = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "HEIF": ".heic",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}

LOSSLESS_OR_UNCOMPRESSED_FORMATS: set[str] = {
    "BMP",
    "GIF",
    "PNG",
    "TIFF",
}

LOSSY_FORMATS: set[str] = {
    "AVIF",
    "HEIF",
    "JPEG",
    "WEBP",
}


def list_images(directories: Pathish | Sequence[Pathish]) -> Iterator[Path]:
    """Yield supported image files from paths or globs."""
    # normalize input to a list of Path-like entries
    if isinstance(directories, (str, os.PathLike)):
        entries = [directories]
    else:
        entries = directories

    for entry in entries:
        path = Path(cast("Any", entry))
        # expand glob entries before walking directories
        if isinstance(entry, str) and any(c in entry for c in "*?[]"):
            candidates = Path().glob(entry)
        elif path.is_dir():
            candidates = path.rglob("*")
        elif path.exists():
            candidates = [path]
        else:
            continue

        for file in candidates:
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                yield file


def find_duplicate_basenames(directory: Pathish) -> dict[Path, list[str]]:
    """Find image stems with multiple extensions."""
    basenames: defaultdict[Path, list[str]] = defaultdict(list)

    for filename in list_images(directory):
        basenames[filename.with_suffix("")].append(filename.suffix)

    duplicates = {
        basename: exts for basename, exts in basenames.items() if len(exts) > 1
    }

    return duplicates


def image_format(filename: Path) -> str:
    """Return Pillow's detected image format."""
    with Image.open(filename) as image:
        detected_format = image.format
    if detected_format is None:
        raise ValueError(f"Could not detect image format for {filename}.")
    return detected_format


def expected_extension(detected_format: str) -> str:
    """Return the preferred extension for an image format."""
    try:
        return IMAGE_FORMAT_EXTENSIONS[detected_format]
    except KeyError as error:
        raise ValueError(f"Unsupported image format {detected_format!r}.") from error


def fix_image_extension(
    filename: Path,
    *,
    dry_run: bool = False,
    verbose: int = 1,
    relative_to: Pathish | None = None,
) -> tuple[Path, str]:
    """Rename an image when its extension mismatches its format."""
    detected_format = image_format(filename)
    expected_formats = EXTENSION_IMAGE_FORMATS.get(filename.suffix.lower(), set())
    if detected_format in expected_formats:
        return filename, detected_format

    target = Path(
        make_unique(filename.with_suffix(expected_extension(detected_format)))
    )
    display_directory = (
        Path(relative_to) if relative_to is not None else filename.parent
    )
    if verbose >= 1:
        print(
            display_file_operation(
                "renaming",
                filename,
                target,
                verbose=verbose,
                relative_to=display_directory,
            ),
            end="",
        )
    try:
        if not dry_run:
            filename.rename(target)
        if verbose >= 1:
            print("success!")
    except Exception:
        if verbose >= 1:
            print("error!")
        raise
    return target, detected_format


def conversion_target(
    filename: Path,
    detected_format: str | None = None,
) -> tuple[Path, str]:
    """Choose a converted filename and Pillow format."""
    if detected_format is None:
        detected_format = image_format(filename)
    if detected_format in LOSSLESS_OR_UNCOMPRESSED_FORMATS:
        output_extension = ".png"
        output_format = "PNG"
    elif detected_format in LOSSY_FORMATS:
        output_extension = ".jpg"
        output_format = "JPEG"
    else:
        raise ValueError(
            f"Unsupported image format {detected_format!r} for {filename}."
        )
    return Path(make_unique(filename.with_suffix(output_extension))), output_format


def convert_images(
    directory: Pathish,
    input_extensions: Sequence[str] = UNWELCOME_EXTENSIONS,
    dry_run: bool = False,
    verbose: int = 1,
) -> None:
    """Convert unwelcome image formats to a target format."""
    display_directory = Path(directory)
    for filename in list_images(directory):
        filename, detected_format = fix_image_extension(
            filename,
            dry_run=dry_run,
            verbose=verbose,
            relative_to=display_directory,
        )
        if expected_extension(detected_format) in input_extensions:
            # choose a collision-free output filename
            output_filename, output_format = conversion_target(
                filename, detected_format
            )

            if verbose >= 1:
                print(
                    display_file_operation(
                        "converting",
                        filename,
                        output_filename,
                        verbose=verbose,
                        relative_to=display_directory,
                    ),
                    end="",
                )
            try:
                if not dry_run:
                    with Image.open(filename) as img:
                        img.convert("RGB").save(output_filename, output_format)
                if verbose >= 1:
                    print("success!")
            except Exception:
                if verbose >= 1:
                    print("error!")
                raise


def delete_duplicate_images(
    directory: Pathish,
    input_extensions: Sequence[str] = UNWELCOME_EXTENSIONS,
    dry_run: bool = False,
    verbose: int = 1,
) -> None:
    """Remove unwelcome duplicates when welcome copies exist."""
    display_directory = Path(directory)
    dupes = find_duplicate_basenames(directory)
    for base, exts in dupes.items():
        welcome_exts = [ext for ext in exts if (ext not in input_extensions)]
        if welcome_exts:
            welcome_filename = base.with_suffix(welcome_exts[0])
            for ext in exts:
                if ext in input_extensions:
                    unwelcome_filename = base.with_suffix(ext)
                    if verbose >= 1:
                        print(
                            display_file_operation(
                                "removing duplicate",
                                unwelcome_filename,
                                welcome_filename,
                                verbose=verbose,
                                relative_to=display_directory,
                            ),
                            end="",
                        )
                    try:
                        if not dry_run:
                            unwelcome_filename.unlink()
                        if verbose >= 1:
                            print("success!")
                    except Exception:
                        if verbose >= 1:
                            print("error!")
                        raise


def normalize_image_extensions(
    directory: Pathish,
    dry_run: bool = False,
    verbose: int = 1,
) -> None:
    """Lowercase supported image extensions."""
    display_directory = Path(directory)
    for filename in list_images(directory):
        if filename.suffix.lower() in IMAGE_EXTENSIONS:
            new_file = filename.with_suffix(filename.suffix.lower())
            if filename != new_file:
                if verbose >= 1:
                    print(
                        display_file_operation(
                            "renaming",
                            filename,
                            new_file,
                            verbose=verbose,
                            relative_to=display_directory,
                        ),
                        end="",
                    )
                try:
                    if not dry_run:
                        filename.rename(new_file)
                    if verbose >= 1:
                        print("success!")
                except Exception:
                    if verbose >= 1:
                        print("error!")
                    raise


def count_files_by_extension(directory: Pathish) -> dict[str, int]:
    """Count supported image files by extension."""
    extension_counts: defaultdict[str, int] = defaultdict(int)

    for filename in list_images(directory):
        extension_counts[filename.suffix] += 1

    return dict(sorted(extension_counts.items()))


def format_extension_counts(extension_counts: dict[str, int]) -> str:
    """Format extension counts for console output."""
    if not extension_counts:
        return "no image files found"
    return "\n".join(
        f"{extension}: {count}" for extension, count in extension_counts.items()
    )


def rename_jpeg_to_jpg(
    directory: Pathish,
    dry_run: bool = False,
    verbose: int = 1,
) -> None:
    """Rename .jpeg files to .jpg."""
    display_directory = Path(directory)
    for filename in list_images(directory):
        if filename.suffix.lower() == ".jpeg":
            filename, detected_format = fix_image_extension(
                filename,
                dry_run=dry_run,
                verbose=verbose,
                relative_to=display_directory,
            )
            if detected_format != "JPEG" or filename.suffix.lower() != ".jpeg":
                continue
            new_file = filename.with_suffix(".jpg")
            if verbose >= 1:
                print(
                    display_file_operation(
                        "renaming",
                        filename,
                        new_file,
                        verbose=verbose,
                        relative_to=display_directory,
                    ),
                    end="",
                )
            try:
                if not dry_run:
                    filename.rename(new_file)
                if verbose >= 1:
                    print("success!")
            except Exception:
                if verbose >= 1:
                    print("error!")
                raise


def main(
    directory: Path,
    dry_run: bool = False,
    verbose: int = 1,
) -> None:
    """Run the full conversion workflow."""
    if verbose == 1:
        print(f"working in {quote_display_path(directory)}")
    find_duplicate_basenames(directory)
    convert_images(directory, dry_run=dry_run, verbose=verbose)
    delete_duplicate_images(directory, dry_run=dry_run, verbose=verbose)
    normalize_image_extensions(directory, dry_run=dry_run, verbose=verbose)
    rename_jpeg_to_jpg(directory, dry_run=dry_run, verbose=verbose)
    if verbose >= 1:
        print(format_extension_counts(count_files_by_extension(directory)))


if __name__ == "__main__":
    main(UPLOAD_DIR)
