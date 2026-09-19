import os
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .constants import REVIEW_ID_COLUMN
from .stackmap import StackMap
from .tagging import ensure_metadata_review_ids, new_metadata_review_id
from .util import Pathish, display_file_operation, make_unique, quote_display_path


def prune_metadata_rows(
    csv_filename: Pathish,
    *,
    verbose: int = 1,
    dry_run: bool = False,
) -> int:
    """Remove metadata rows whose image files are no longer present."""
    csv_path = Path(csv_filename)
    ensure_metadata_review_ids(csv_path)
    if not csv_path.is_file():
        if verbose >= 1:
            print(f"no metadata file: {csv_path.name}")
        return 0

    try:
        metadata_df = pd.read_csv(csv_path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        if not dry_run:
            csv_path.unlink(missing_ok=True)
        if verbose >= 1:
            print(f"removed 0 row(s) from {csv_path.name}")
        return 0

    rows_to_keep: list[dict[str, Any]] = []
    removed_count = 0

    for raw_item in metadata_df.to_dict("records"):
        item = cast("dict[str, Any]", dict(raw_item))
        original_path = str(item.get("original_filepath", "")).strip()
        clean_filename = str(item.get("clean_filename", "")).strip()
        candidate_paths = [Path(original_path)] if original_path else []
        if original_path and clean_filename:
            candidate_paths.append(Path(original_path).with_name(clean_filename))
        if any(path.is_file() for path in candidate_paths):
            rows_to_keep.append(item)
            continue
        if verbose >= 2:
            print(f"removing row: {original_path}")
        removed_count += 1

    if not dry_run:
        if rows_to_keep:
            pd.DataFrame(rows_to_keep, columns=metadata_df.columns).to_csv(
                csv_path,
                index=False,
            )
        else:
            csv_path.unlink(missing_ok=True)

    if verbose >= 1:
        print(f"removed {removed_count} row(s) from {csv_path.name}")
    return removed_count


def rename_images(
    csv_filename: Pathish,
    verbose: int = 1,
    dry_run: bool = False,
) -> None:
    """Rename images from metadata suggestions."""
    csv_path = Path(csv_filename)
    ensure_metadata_review_ids(csv_path)
    metadata_df = pd.read_csv(csv_path)
    metadata_updated = False
    display_directory = csv_path.parent
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for index, row in metadata_df.iterrows():
        source = Path(row["original_filepath"])

        if row["status"] != "ok" or not row["clean_filename"]:
            if verbose >= 2:
                print(f"skipping errored row {index} {source!r}")
            continue

        new_filename = row["clean_filename"]
        target = source.with_name(new_filename)

        if not source.is_file():
            if verbose >= 2:
                print(f"source file {os.fspath(source)!r} is missing!")
            if verbose >= 1 and not target.is_file():
                print(
                    f"both source file {os.fspath(source)!r} and {os.fspath(target)!r} are missing!"
                )
            continue

        if target == source:
            if verbose >= 2:
                print(f"no rename necessary for {source!r}")
            continue

        if source.suffix.lower() != target.suffix.lower():
            if verbose >= 1:
                print(
                    f"Mismatched file extensions between {os.fspath(source)!r} and {os.fspath(target)!r}; skipping rename!"
                )
            continue

        if target.is_file():
            if verbose >= 1:
                print(f"target {os.fspath(target)!r} already exists!")
            target = Path(make_unique(target))
            if verbose >= 1:
                print(f"proceeding with target {os.fspath(target)!r}.")

        if verbose >= 1:
            print(
                display_file_operation(
                    "renaming",
                    source,
                    target,
                    verbose=verbose,
                    relative_to=display_directory,
                ),
                end="",
            )
        try:
            if not dry_run:
                source.rename(target)
                target_filename = target.name
                if row["clean_filename"] != target_filename:
                    metadata_df.at[index, "clean_filename"] = target_filename
                    metadata_updated = True
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            else:
                print(f"error renaming {os.fspath(source)!r} to {os.fspath(target)!r}!")

    if metadata_updated:
        metadata_df.to_csv(csv_path, index=False)


def append_shelved_metadata(
    row: pd.Series[Any],
    target: Path,
    source_metadata_path: Path,
) -> None:
    """Append shelved image metadata to the target directory metadata file."""
    target_metadata_path = target.parent / source_metadata_path.name
    ensure_metadata_review_ids(target_metadata_path)
    if not target_metadata_path.is_file():
        return

    target_metadata_df = pd.read_csv(target_metadata_path)
    target_filenames = set()
    for column in ["original_filename", "clean_filename"]:
        if column in target_metadata_df:
            target_filenames.update(
                Path(value).name
                for value in target_metadata_df[column].dropna()
                if value
            )
    if target.name in target_filenames:
        return

    target_row = row.copy()
    target_row["original_filepath"] = os.fspath(target)
    target_row["original_filename"] = target.name
    target_row["clean_filename"] = target.name
    target_row = target_row.reindex(target_metadata_df.columns)
    if REVIEW_ID_COLUMN in target_row:
        target_row[REVIEW_ID_COLUMN] = new_metadata_review_id()
    target_row.to_frame().T.to_csv(
        target_metadata_path,
        mode="a",
        header=False,
        index=False,
    )


def _stackmap_alias_for_directory(stackmap: StackMap, directory: Path) -> str | None:
    """Return the configured shelf alias for a directory, if any."""
    resolved_directory = directory.resolve()
    for alias, shelf_directory in stackmap.shelves.items():
        if shelf_directory == resolved_directory:
            return alias
    return None


def shelve_images(
    csv_filename: Pathish,
    stackmap: StackMap,
    verbose: int = 1,
    dry_run: bool = False,
    review_ids: set[str] | None = None,
    prune_verbose: int | None = None,
) -> None:
    """Move images into category folders, while keeping the current shelf scoped."""
    csv_path = Path(csv_filename)
    ensure_metadata_review_ids(csv_path)
    if not csv_path.is_file():
        return

    metadata_df = pd.read_csv(csv_path)
    display_directory = stackmap.filename.parent
    current_alias = _stackmap_alias_for_directory(stackmap, csv_path.parent)
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for position, (index, row) in enumerate(metadata_df.iterrows(), start=1):
        review_id = str(row.get(REVIEW_ID_COLUMN, position))
        if review_ids is not None and review_id not in review_ids:
            continue
        original_source = Path(row["original_filepath"])

        if row["status"] != "ok" or not row["category"]:
            if verbose >= 2:
                print(f"skipping row {index} {original_source!r}")
            continue

        source_filename = row["clean_filename"] or row["original_filename"]
        source = original_source.with_name(source_filename)
        if not source.is_file():
            source = original_source

        if not source.is_file():
            if verbose >= 1:
                print(f"source file {os.fspath(source)!r} is missing!")
            continue

        category = str(row["category"]).strip()
        if category == "default" or category not in stackmap.shelves:
            if verbose >= 1:
                print(
                    f"category {category!r} is not a configured shelf; skipping {source!r}"
                )
            continue
        if current_alias is not None and category == current_alias:
            if verbose >= 2:
                print(f"already in correct shelf {category!r}; skipping {source!r}")
            continue

        target_directory = stackmap.directory_for(category)
        if target_directory is None:
            if verbose >= 1:
                print(
                    f"category {category!r} is not a configured shelf; skipping {source!r}"
                )
            continue
        if not target_directory.is_dir():
            if verbose >= 1:
                print(
                    f"target directory {os.fspath(target_directory)!r} is missing; skipping {os.fspath(source)!r}"
                )
            continue

        target = target_directory / source.name

        if target == source:
            if verbose >= 2:
                print(f"no move necessary for {source!r}")
            continue

        if target.is_file():
            if verbose >= 1:
                print(f"target {os.fspath(target)!r} already exists!")
            target = Path(make_unique(target))
            if verbose >= 1:
                print(f"proceeding with target {os.fspath(target)!r}.")

        if verbose >= 1:
            print(
                display_file_operation(
                    "moving",
                    source,
                    target,
                    verbose=verbose,
                    relative_to=display_directory,
                ),
                end="",
            )
        try:
            if not dry_run:
                source.rename(target)
                append_shelved_metadata(row, target, csv_path)
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            else:
                print(f"error moving {os.fspath(source)!r} to {os.fspath(target)!r}!")

    if not dry_run:
        prune_metadata_rows(
            csv_path,
            verbose=verbose if prune_verbose is None else prune_verbose,
        )
