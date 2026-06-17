import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from constants import IMAGE_EXTENSIONS, UNWELCOME_EXTENSIONS, UPLOAD_DIR
from util import make_unique

register_heif_opener()


def list_images(directories: str | Path | Sequence[str | Path]) -> Iterator[Path]:
    """Yield supported image files from paths or globs."""
    # normalize input to a list of Path-like entries
    if isinstance(directories, (str, Path)):
        directories = [directories]

    for entry in directories:
        path = Path(entry)
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


def find_duplicate_basenames(directory: str | Path) -> dict[str, list[str]]:
    """Find image stems with multiple extensions."""
    basenames: defaultdict[str, list[str]] = defaultdict(list)

    for filename in list_images(directory):
        basename, ext = os.path.splitext(filename)
        basenames[basename].append(ext)

    duplicates = {
        basename: exts for basename, exts in basenames.items() if len(exts) > 1
    }

    return duplicates


def convert_images(
    directory: str | Path,
    input_extensions: Sequence[str] = UNWELCOME_EXTENSIONS,
    output_format: str = "JPEG",
    output_extension: str = ".jpg",
    dry_run: bool = False,
) -> None:
    """Convert unwelcome image formats to a target format."""
    for filename in list_images(directory):
        file_path = os.path.join(directory, filename)
        base_name, extension = os.path.splitext(filename)

        if extension.lower() in input_extensions:
            # choose a collision-free output filename
            output_filename = os.path.join(directory, base_name + output_extension)
            output_filename = make_unique(output_filename)

            # rewrite the image unless this is a dry run
            if not dry_run:
                with Image.open(file_path) as img:
                    img.convert("RGB").save(output_filename, output_format)
            print(f"Converted {filename} to {output_filename}")


def delete_duplicate_images(
    directory: str | Path,
    input_extensions: Sequence[str] = UNWELCOME_EXTENSIONS,
    dry_run: bool = False,
) -> None:
    """Remove unwelcome duplicates when welcome copies exist."""
    dupes = find_duplicate_basenames(directory)
    for base, exts in dupes.items():
        welcome_exts = [ext for ext in exts if (ext not in input_extensions)]
        if welcome_exts:
            welcome_filename = base + welcome_exts[0]
            for ext in exts:
                if ext in input_extensions:
                    unwelcome_filename = base + ext
                    if not dry_run:
                        os.remove(os.path.join(directory, unwelcome_filename))
                    print(
                        f"removed duplicate file {unwelcome_filename} (retaining {welcome_filename})"
                    )


def normalize_image_extensions(
    directory: str | Path,
    dry_run: bool = False,
) -> None:
    """Lowercase supported image extensions."""
    for filename in list_images(directory):
        base, extension = os.path.splitext(filename)
        if extension.lower() in IMAGE_EXTENSIONS:
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, base + extension.lower())
            if old_file != new_file:
                if not dry_run:
                    os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")


def count_files_by_extension(directory: str | Path) -> dict[str, int]:
    """Count supported image files by extension."""
    extension_counts: defaultdict[str, int] = defaultdict(int)

    for filename in list_images(directory):
        _, extension = os.path.splitext(filename)
        extension_counts[extension] += 1

    return dict(sorted(extension_counts.items()))


def format_extension_counts(extension_counts: dict[str, int]) -> str:
    """Format extension counts for console output."""
    if not extension_counts:
        return "no image files found"
    return "\n".join(
        f"{extension}: {count}" for extension, count in extension_counts.items()
    )


def rename_jpeg_to_jpg(
    directory: str | Path,
    dry_run: bool = False,
) -> None:
    """Rename .jpeg files to .jpg."""
    for filename in list_images(directory):
        base, extension = os.path.splitext(filename)
        if extension.lower() == ".jpeg":
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, base + ".jpg")
            if not dry_run:
                os.rename(old_file, new_file)
            print(f"Renamed '{old_file}' to '{new_file}'")


def main(
    directory: Path,
    dry_run: bool = False,
) -> None:
    """Run the full conversion workflow."""
    directory_string = str(directory)
    find_duplicate_basenames(directory_string)
    convert_images(directory_string, dry_run=dry_run)
    delete_duplicate_images(directory_string, dry_run=dry_run)
    normalize_image_extensions(directory, dry_run=dry_run)
    rename_jpeg_to_jpg(directory, dry_run=dry_run)
    print(format_extension_counts(count_files_by_extension(directory)))


if __name__ == "__main__":
    main(UPLOAD_DIR)
