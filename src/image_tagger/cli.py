import argparse
import re
from pathlib import Path

from . import review_app, transform
from .compare import prune_embedding_cache
from .constants import (
    DEDUPE_EMBEDDINGS_FILENAME,
    DEDUPE_REVIEW_FILENAME,
    DEFAULT_AUTOMATIC_THRESHOLD,
    DEFAULT_LARGE_IMAGE_THRESHOLD,
    DEFAULT_LLM_THRESHOLD,
    DEFAULT_WALL_ASSUMED_COLUMNS,
    DEFAULT_WALL_RANDOM_SEED,
    GALLERY_NAME,
    IMAGE_EXTENSIONS,
    METADATA_FILENAME,
    WALL_DOUBLE_WIDE_THRESHOLD,
    WELCOME_EXTENSIONS,
)
from .convert import (
    convert_images,
    count_files_by_extension,
    delete_duplicate_images,
    format_extension_counts,
    normalize_image_extensions,
    rename_jpeg_to_jpg,
)
from .dedupe import dedupe_images
from .gallery import generate_gallery
from .report import report_images
from .shelve import prune_metadata_rows, rename_images, shelve_images
from .stackmap import StackMap, find_stackmap
from .tagging import (
    ensure_metadata_review_ids,
    find_images,
    populate_metadata_quads,
    tag_images,
    update_metadata_filepaths,
)
from .util import preview, quote_display_path
from .util import scramble as scramble_name
from .vision import VisionModelProvider
from .wall import generate_wall


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


def file_size_arg(value: str) -> int:
    """Parse a lenient decimal file size such as 1 MB or 500k."""
    normalized_value = value.strip().lower().replace(",", "").replace("_", "")
    match = re.fullmatch(
        r"(\d+(?:\.\d*)?|\.\d+)\s*([kmgt]?)\s*(?:i?b(?:ytes?)?)?",
        normalized_value,
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            f"Invalid file size {value!r}; use a value such as 1 MB or 500k."
        )
    amount = float(match.group(1))
    multipliers = {
        "": 1,
        "k": 1_000,
        "m": 1_000_000,
        "g": 1_000_000_000,
        "t": 1_000_000_000_000,
    }
    size = int(amount * multipliers[match.group(2)])
    if size <= 0:
        raise argparse.ArgumentTypeError("File size must be greater than zero.")
    return size


def positive_int_arg(value: str) -> int:
    """Parse a positive integer argument."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed_value


def convert(args: argparse.Namespace) -> None:
    """Run upload conversion steps."""
    directory = args.directory
    unwelcome_extensions = [
        extension
        for extension in IMAGE_EXTENSIONS
        if extension not in args.welcome_extensions
    ]
    if args.verbose == 1:
        print(f"working in {quote_display_path(directory)}")
    file_changes = convert_images(
        directory,
        input_extensions=unwelcome_extensions,
        welcome_extensions=args.welcome_extensions,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    file_changes.extend(
        delete_duplicate_images(
            directory,
            input_extensions=unwelcome_extensions,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    )
    file_changes.extend(
        normalize_image_extensions(
            directory,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    )
    file_changes.extend(
        rename_jpeg_to_jpg(
            directory,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    )
    if not args.dry_run:
        update_metadata_filepaths(args.metadata_filename, file_changes)
    if args.verbose >= 1:
        print(format_extension_counts(count_files_by_extension(directory)))


def tag(args: argparse.Namespace) -> None:
    """Tag upload images."""
    filepaths = find_images(
        args.directory,
        metadata_filename=args.metadata_filename,
        extension_filter=args.extensions,
    )
    print("number of image files to tag:", len(filepaths))
    tag_images(
        filepaths,
        args.metadata_filename,
        retry_errors=args.retry_errors,
        verbose=args.verbose,
        provider=args.provider,
        instructions_filename=args.instructions_filename,
        categories=args.stackmap_config.categories,
        category_descriptions=args.stackmap_config.category_descriptions,
        quad_detector=transform.detect_quad if args.quad else None,
    )


def quad(args: argparse.Namespace) -> None:
    """Populate missing perspective-corner metadata for tagged upload images."""
    count = populate_metadata_quads(
        args.metadata_filename,
        quad_detector=transform.detect_quad,
        provider=args.provider,
        verbose=args.verbose,
    )
    print("number of tagged image files quad calculated:", count)


def dedupe(args: argparse.Namespace) -> None:
    """Remove duplicate upload images."""
    dedupe_images(
        args.directory,
        automatic_threshold=args.automatic_threshold,
        llm_threshold=args.llm_threshold,
        verbose=args.verbose,
        dry_run=args.dry_run,
        provider=args.provider,
        filename_glob=args.filename,
    )
    review_filename = args.directory / DEDUPE_REVIEW_FILENAME
    if args.preview and review_filename.is_file():
        preview(review_filename)


def crop(args: argparse.Namespace) -> None:
    """Perspective-correct images with one CV-assisted VLM pass."""
    transform.transform_images(
        args.directory,
        provider=VisionModelProvider(args.provider),
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    review_filename = args.directory / transform.TRANSFORM_REVIEW_FILENAME
    if args.preview and review_filename.is_file():
        preview(review_filename)


def prune(args: argparse.Namespace) -> None:
    """Remove rows whose image files are missing from the metadata CSV."""
    prune_metadata_rows(
        args.metadata_filename,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        prune_embedding_cache(args.directory / DEDUPE_EMBEDDINGS_FILENAME)


def rename(args: argparse.Namespace) -> None:
    """Rename uploads from metadata."""
    rename_images(
        args.metadata_filename,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def shelve(args: argparse.Namespace) -> None:
    """Move uploads into category folders."""
    shelve_images(
        args.metadata_filename,
        stackmap=args.stackmap_config,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def scramble(args: argparse.Namespace) -> None:
    """Scramble upload filenames."""
    renamed_count = 0
    for source in find_images(args.directory):
        stem = source.stem
        extension = source.suffix
        target = args.directory / f"{scramble_name(stem)}{extension}"

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


def gallery(args: argparse.Namespace) -> None:
    """Generate and optionally preview a gallery."""
    output_filename = (
        args.output_filename
        if args.output_filename is not None
        else args.directory / GALLERY_NAME.name
    )
    generate_gallery(
        args.metadata_filename,
        output_filename,
        verbose=args.verbose,
    )
    if args.preview:
        preview(output_filename)


def wall(args: argparse.Namespace) -> None:
    """Generate and optionally preview an image wall."""
    output_filename = generate_wall(
        args.directory,
        args.output_filename,
        metadata_filename=args.metadata_filename,
        order=args.order,
        seed=args.seed,
        title=args.title,
        double_wide_threshold=args.double_wide_threshold,
        assume_columns=args.assume_columns,
        verbose=args.verbose,
    )
    if args.preview:
        preview(output_filename)


def report(args: argparse.Namespace) -> None:
    """Print an image collection report."""
    print(
        report_images(
            args.directory,
            args.metadata_filename,
            args.large_image_threshold,
        )
    )


def review(args: argparse.Namespace) -> None:
    """Serve an editable metadata review app."""
    review_app.review_metadata(
        args.metadata_filename,
        stackmap=args.stackmap_config,
        provider=args.provider,
        verbose=args.verbose,
        filename_glob=args.filename,
    )


def run(args: argparse.Namespace) -> None:
    """Run the default upload workflow sequence."""
    convert(args)
    tag(args)
    rename(args)
    quad(args)
    review(args)


def clean(args: argparse.Namespace) -> None:
    """Remove generated workflow files."""
    ensure_metadata_review_ids(args.metadata_filename)
    for filename in [
        args.metadata_filename,
        args.metadata_filename.with_suffix(f"{args.metadata_filename.suffix}.bak"),
        args.output_filename,
        args.directory / DEDUPE_REVIEW_FILENAME,
        args.directory / DEDUPE_EMBEDDINGS_FILENAME,
    ]:
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
    parser.add_argument("directory", nargs="?", type=path_arg)
    parser.add_argument(
        "--stackmap",
        type=path_arg,
        help="Path to the library shelf map. Defaults to a discovered .stackmap file.",
    )
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
    convert_parser.add_argument(
        "-w",
        "--welcome-extensions",
        type=extensions_arg,
        default=WELCOME_EXTENSIONS,
        help="Comma-delimited image extensions to preserve.",
    )
    convert_parser.set_defaults(func=convert)

    dedupe_parser = subparsers.add_parser(
        "dedupe",
        help="Remove duplicate uploads with CLIP and optional LLM checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_upload_args(dedupe_parser)
    dedupe_parser.add_argument(
        "--automatic-threshold",
        type=float,
        default=DEFAULT_AUTOMATIC_THRESHOLD,
        help="CLIP score accepted as duplicate without LLM confirmation.",
    )
    dedupe_parser.add_argument(
        "--llm-threshold",
        type=float,
        default=DEFAULT_LLM_THRESHOLD,
        help="CLIP score sent to the LLM for duplicate confirmation.",
    )
    dedupe_parser.add_argument(
        "--filename",
        help="Check only pairs where at least one filename matches this glob.",
    )
    dedupe_parser.add_argument(
        "--provider",
        choices=["openai", "gemma", "qwen"],
        default="openai",
        help="Vision model provider for borderline duplicate confirmation.",
    )
    dedupe_parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    dedupe_parser.set_defaults(func=dedupe)

    transform_parser = subparsers.add_parser(
        "crop",
        aliases=["clip"],
        help="Perspective-correct rectangular images with a vision model and OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_upload_args(transform_parser)
    transform_parser.add_argument(
        "--provider",
        choices=["openai", "gemma", "qwen"],
        default="openai",
        help="Vision model provider for crop adjudication.",
    )
    transform_parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    transform_parser.set_defaults(func=crop)

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
    tag_parser.add_argument(
        "--quad",
        action="store_true",
        help="Detect and store normalized perspective-corner coordinates.",
    )
    tag_parser.set_defaults(func=tag)

    quad_parser = subparsers.add_parser(
        "quad",
        help="Populate missing perspective-corner metadata for tagged images.",
    )
    add_common_upload_args(quad_parser)
    quad_parser.add_argument(
        "--provider",
        choices=["openai", "gemma", "qwen"],
        default="openai",
    )
    quad_parser.set_defaults(func=quad)

    prune_parser = subparsers.add_parser(
        "prune", help="Remove metadata rows whose image files are missing."
    )
    add_common_upload_args(prune_parser)
    prune_parser.set_defaults(func=prune)

    rename_parser = subparsers.add_parser(
        "rename", help="Rename uploads from metadata suggestions."
    )
    add_common_upload_args(rename_parser)
    rename_parser.set_defaults(func=rename)

    shelve_parser = subparsers.add_parser(
        "shelve", help="Move uploads into configured category shelves."
    )
    add_common_upload_args(shelve_parser)
    shelve_parser.set_defaults(func=shelve)

    scramble_parser = subparsers.add_parser(
        "scramble", help="Randomize upload filename stems in place."
    )
    add_common_upload_args(scramble_parser)
    scramble_parser.set_defaults(func=scramble)

    gallery_parser = subparsers.add_parser(
        "gallery", help="Generate the upload gallery HTML."
    )
    add_common_upload_args(gallery_parser)
    gallery_parser.add_argument("--output-filename", type=path_arg)
    gallery_parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    gallery_parser.set_defaults(func=gallery)

    review_parser = subparsers.add_parser(
        "review", help="Serve an editable metadata review app."
    )
    add_common_upload_args(review_parser)
    review_parser.add_argument(
        "--provider",
        choices=["openai", "gemma", "qwen"],
        default="openai",
        help="Vision model provider for interactive crop review.",
    )
    review_parser.add_argument(
        "--filename",
        help="Review only images whose filename matches this glob.",
    )
    review_parser.set_defaults(func=review)

    wall_parser = subparsers.add_parser(
        "wall", help="Generate a full-window image wall HTML page."
    )
    add_common_upload_args(wall_parser)
    wall_parser.add_argument("--output-filename", type=path_arg)
    wall_parser.add_argument(
        "--order",
        choices=["name", "date", "random", "grid"],
        default="random",
        help="Order wall images by name, newest date first, random shuffle (default), or CLIP similarity grid.",
    )
    wall_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_WALL_RANDOM_SEED,
        help=f"Seed for deterministic random wall ordering. Defaults to {DEFAULT_WALL_RANDOM_SEED}.",
    )
    wall_parser.add_argument(
        "--title",
        help="HTML page title for the generated wall.",
    )
    wall_parser.add_argument(
        "--double-wide-threshold",
        type=float,
        default=WALL_DOUBLE_WIDE_THRESHOLD,
        help="Mark images this many times wider than a cell as double-wide.",
    )
    wall_parser.add_argument(
        "--assume-columns",
        type=positive_int_arg,
        default=DEFAULT_WALL_ASSUMED_COLUMNS,
        help=f"Assume this many columns when computing grid order. Defaults to {DEFAULT_WALL_ASSUMED_COLUMNS}.",
    )
    wall_parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    wall_parser.set_defaults(func=wall)

    report_parser = subparsers.add_parser(
        "report", help="Print an image collection report."
    )
    add_common_upload_args(report_parser)
    report_parser.add_argument(
        "-s",
        "--large-image-threshold",
        type=file_size_arg,
        default=DEFAULT_LARGE_IMAGE_THRESHOLD,
        help="Show only images larger than this size, such as 1 MB or 500k.",
    )
    report_parser.set_defaults(func=report)

    run_parser = subparsers.add_parser(
        "run",
        help="Run convert, tag, rename, quad, and review in sequence.",
    )
    add_common_upload_args(run_parser)
    run_parser.add_argument(
        "-w",
        "--welcome-extensions",
        type=extensions_arg,
        default=WELCOME_EXTENSIONS,
        help="Comma-delimited image extensions to preserve during convert.",
    )
    run_parser.add_argument(
        "--provider",
        choices=["openai", "gemma", "qwen"],
        default="openai",
        help="Vision model provider for tag, quad, and review steps.",
    )
    run_parser.add_argument(
        "--extensions",
        type=extensions_arg,
        default=WELCOME_EXTENSIONS,
        help="Comma-delimited image extensions to include during tagging.",
    )
    run_parser.add_argument(
        "--instructions-filename",
        type=path_arg,
        help="Optional prompt template for tagging.",
    )
    run_parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry metadata rows whose status is error during tagging.",
    )
    run_parser.add_argument(
        "--quad",
        action="store_true",
        help="Also detect perspective-corner metadata during tagging.",
    )
    run_parser.add_argument(
        "--filename",
        help="Review only images whose filename matches this glob.",
    )
    run_parser.set_defaults(func=run)

    clean_parser = subparsers.add_parser(
        "clean", help="Remove generated metadata and gallery files."
    )
    add_common_upload_args(clean_parser)
    clean_parser.add_argument("--output-filename", type=path_arg)
    clean_parser.set_defaults(func=clean)

    return parser


def main() -> None:
    """Run the selected CLI command."""
    parser = build_parser()
    args = parser.parse_args()
    stackmap_filename = args.stackmap or find_stackmap()
    if stackmap_filename is None:
        raise SystemExit("could not find .stackmap; pass --stackmap to specify one.")
    try:
        args.stackmap_config = StackMap.load(stackmap_filename)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.directory is None:
        args.directory = args.stackmap_config.default_directory
    else:
        args.directory = (
            args.stackmap_config.directory_for(str(args.directory)) or args.directory
        )
    if args.metadata_filename is None:
        args.metadata_filename = args.directory / METADATA_FILENAME.name
    if args.command in {
        "convert",
        "tag",
        "quad",
        "prune",
        "rename",
        "shelve",
        "review",
        "clean",
        "run",
    }:
        ensure_metadata_review_ids(args.metadata_filename)
    if args.command == "clean" and args.output_filename is None:
        args.output_filename = args.directory / GALLERY_NAME.name
    args.verbose += args.verbose_delta - args.quiet_delta
    args.func(args)


if __name__ == "__main__":
    main()
