import argparse
import re
from pathlib import Path

import image_tagger as it
from constants import GALLERY_NAME, METADATA_FILENAME, UPLOAD_DIR, WELCOME_EXTENSIONS
from convert import (
    convert_images,
    count_files_by_extension,
    delete_duplicate_images,
    format_extension_counts,
    normalize_image_extensions,
    rename_jpeg_to_jpg,
)
from util import preview, quote_display_path


def path_arg(value: str) -> Path:
    """Parse a CLI path argument."""
    return Path(value)


def extensions_arg(value: str) -> list[str]:
    """Parse comma, semicolon, or whitespace-separated extensions."""
    extensions: list[str] = []
    for extension in re.split(r"[\s,;]+", value.strip()):
        if not extension:
            continue
        extension = extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return extensions


def convert_uploads(args: argparse.Namespace) -> None:
    """Run upload conversion steps."""
    directory = args.directory
    if args.verbose == 1:
        print(f"working in {quote_display_path(directory)}")
    convert_images(directory, dry_run=args.dry_run, verbose=args.verbose)
    delete_duplicate_images(directory, dry_run=args.dry_run, verbose=args.verbose)
    normalize_image_extensions(directory, dry_run=args.dry_run, verbose=args.verbose)
    rename_jpeg_to_jpg(directory, dry_run=args.dry_run, verbose=args.verbose)
    if args.verbose >= 1:
        print(format_extension_counts(count_files_by_extension(directory)))


def tag_uploads(args: argparse.Namespace) -> None:
    """Tag upload images."""
    filepaths = it.find_images(
        args.directory,
        metadata_filename=args.metadata_filename,
        extension_filter=args.extensions,
    )
    print("number of image files to tag:", len(filepaths))
    it.tag_images(
        filepaths,
        args.metadata_filename,
        retry_errors=args.retry_errors,
        verbose=args.verbose,
        provider=args.provider,
        instructions_filename=args.instructions_filename,
    )


def rename_uploads(args: argparse.Namespace) -> None:
    """Rename uploads from metadata."""
    it.rename_images(
        args.metadata_filename,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def shelve_uploads(args: argparse.Namespace) -> None:
    """Move uploads into category folders."""
    it.shelve_images(
        args.metadata_filename,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def scramble_uploads(args: argparse.Namespace) -> None:
    """Scramble upload filenames."""
    renamed_count = 0
    for source in it.find_images(args.directory):
        stem = source.stem
        extension = source.suffix
        target = args.directory / f"{it.scramble(stem)}{extension}"

        if source == target:
            continue

        if target.exists():
            print(f"Skipping {source}; target already exists: {target}")
            continue

        if not args.dry_run:
            source.rename(target)
        renamed_count += 1
        print(f"Renamed {source.name} -> {target.name}")

    print(f"Renamed {renamed_count} files in {args.directory}")


def gallery_uploads(args: argparse.Namespace) -> None:
    """Generate and optionally preview a gallery."""
    it.generate_gallery(
        args.metadata_filename,
        args.output_filename,
        verbose=args.verbose,
    )
    if args.preview:
        preview(args.output_filename)


def wall_uploads(args: argparse.Namespace) -> None:
    """Generate and optionally preview an image wall."""
    output_filename = it.generate_wall(
        args.directory,
        args.output_filename,
        metadata_filename=args.metadata_filename,
        order=args.order,
        verbose=args.verbose,
    )
    if args.preview:
        preview(output_filename)


def clean_uploads(args: argparse.Namespace) -> None:
    """Remove generated workflow files."""
    for filename in [args.metadata_filename, args.output_filename]:
        if filename.exists():
            if args.dry_run:
                print(f"Would remove {filename}")
            else:
                filename.unlink()
                print(f"Removed {filename}")
        elif args.verbose >= 2:
            print(f"Already clean: {filename}")


def add_common_upload_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by upload commands."""
    parser.add_argument("directory", nargs="?", type=path_arg, default=UPLOAD_DIR)
    parser.add_argument("--metadata-filename", type=path_arg)
    parser.set_defaults(verbose=1)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose_delta",
        help="Increase verbosity. Repeat for more output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        dest="quiet_delta",
        help="Decrease verbosity. Repeat for less output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="simulate running the command without taking actions.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Image tagger upload workflow tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert uploads to preferred formats."
    )
    add_common_upload_args(convert_parser)
    convert_parser.set_defaults(func=convert_uploads)

    tag_parser = subparsers.add_parser(
        "tag", help="Tag upload images with a vision model."
    )
    add_common_upload_args(tag_parser)
    tag_parser.add_argument(
        "--provider",
        choices=["openai", "gemma", "qwen"],
        default="openai",
    )
    tag_parser.add_argument(
        "--extensions", type=extensions_arg, default=WELCOME_EXTENSIONS
    )
    tag_parser.add_argument("--instructions-filename", type=path_arg)
    tag_parser.add_argument("--retry-errors", action="store_true")
    tag_parser.set_defaults(func=tag_uploads)

    rename_parser = subparsers.add_parser(
        "rename", help="Rename uploads from metadata suggestions."
    )
    add_common_upload_args(rename_parser)
    rename_parser.set_defaults(func=rename_uploads)

    shelve_parser = subparsers.add_parser(
        "shelve", help="Move uploads into sibling category directories."
    )
    add_common_upload_args(shelve_parser)
    shelve_parser.set_defaults(func=shelve_uploads)

    scramble_parser = subparsers.add_parser(
        "scramble", help="Randomize upload filename stems in place."
    )
    add_common_upload_args(scramble_parser)
    scramble_parser.set_defaults(func=scramble_uploads)

    gallery_parser = subparsers.add_parser(
        "gallery", help="Generate the upload gallery HTML."
    )
    add_common_upload_args(gallery_parser)
    gallery_parser.add_argument(
        "--output-filename", type=path_arg, default=GALLERY_NAME
    )
    gallery_parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    gallery_parser.set_defaults(func=gallery_uploads)

    wall_parser = subparsers.add_parser(
        "wall", help="Generate a full-window image wall HTML page."
    )
    add_common_upload_args(wall_parser)
    wall_parser.add_argument("--output-filename", type=path_arg)
    wall_parser.add_argument(
        "--order",
        choices=["name", "date", "random"],
        default="name",
        help="Order wall images by name, newest date first, or random shuffle.",
    )
    wall_parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    wall_parser.set_defaults(func=wall_uploads)

    clean_parser = subparsers.add_parser(
        "clean", help="Remove generated metadata and gallery files."
    )
    add_common_upload_args(clean_parser)
    clean_parser.add_argument("--output-filename", type=path_arg, default=GALLERY_NAME)
    clean_parser.set_defaults(func=clean_uploads)

    return parser


def main() -> None:
    """Run the selected CLI command."""
    parser = build_parser()
    args = parser.parse_args()
    if args.metadata_filename is None:
        args.metadata_filename = args.directory / METADATA_FILENAME.name
    args.verbose += args.verbose_delta - args.quiet_delta
    args.func(args)


if __name__ == "__main__":
    main()
