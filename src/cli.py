import argparse
from pathlib import Path

import image_tagger as it
from constants import GALLERY_NAME, METADATA_FILENAME, UPLOAD_DIR
from convert_uploads import (
    convert_images,
    count_files_by_extension,
    delete_duplicate_images,
    normalize_image_extensions,
    rename_jpeg_to_jpg,
)
from util import preview


def path_arg(value: str) -> Path:
    return Path(value)


def convert_uploads(args: argparse.Namespace) -> None:
    directory = args.directory
    convert_images(str(directory), dry_run=args.dry_run)
    delete_duplicate_images(str(directory), dry_run=args.dry_run)
    normalize_image_extensions(directory, dry_run=args.dry_run)
    rename_jpeg_to_jpg(directory, dry_run=args.dry_run)
    print(count_files_by_extension(directory))


def tag_uploads(args: argparse.Namespace) -> None:
    filepaths = it.find_images(str(args.directory))
    print("number of image files:", len(filepaths))
    it.tag_images(
        filepaths,
        args.metadata_filename,
        retry_errors=args.retry_errors,
        verbose=args.verbose,
        provider=args.provider,
    )


def rename_uploads(args: argparse.Namespace) -> None:
    it.autorename(
        args.metadata_filename,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def scramble_uploads(args: argparse.Namespace) -> None:
    renamed_count = 0
    for filepath in it.find_images(str(args.directory)):
        _, stem, extension = it.path_name_ext(filepath)
        source = args.directory / f"{stem}{extension}"
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
    it.generate_gallery(
        args.metadata_filename, args.output_filename, verbose=args.verbose
    )
    if args.preview:
        preview(args.output_filename)


def clean_uploads(args: argparse.Namespace) -> None:
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
    parser.add_argument("--directory", type=path_arg, default=UPLOAD_DIR)
    parser.add_argument("--metadata-filename", type=path_arg, default=METADATA_FILENAME)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
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
        "--provider", choices=["openai", "gemma", "qwen"], default="openai"
    )
    tag_parser.add_argument("--retry-errors", action="store_true")
    tag_parser.set_defaults(func=tag_uploads)

    rename_parser = subparsers.add_parser(
        "rename", help="Rename uploads from metadata suggestions."
    )
    add_common_upload_args(rename_parser)
    rename_parser.set_defaults(func=rename_uploads)

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

    clean_parser = subparsers.add_parser(
        "clean", help="Remove generated metadata and gallery files."
    )
    add_common_upload_args(clean_parser)
    clean_parser.add_argument("--output-filename", type=path_arg, default=GALLERY_NAME)
    clean_parser.set_defaults(func=clean_uploads)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
