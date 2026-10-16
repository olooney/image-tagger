import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_tagger import cli
from image_tagger.compare import (
    EmbeddingCache,
    _load_embedding_cache,
    _save_embedding_cache,
)
from image_tagger.constants import CLIP_MODEL, DEDUPE_EMBEDDINGS_FILENAME
from image_tagger.report import report_images
from image_tagger.thumbnail import create_thumbnails


def test_create_thumbnails_resizes_images_and_copies_publish_files(
    tmp_path: Path,
) -> None:
    """Create a metadata-free, size-limited copy of an image collection."""
    source = tmp_path / "books"
    destination = tmp_path / "thumbs"
    nested = source / "nested"
    nested.mkdir(parents=True)

    portrait = Image.new("RGB", (800, 1600), "red")
    exif = portrait.getexif()
    exif[315] = "Private photographer"
    portrait.save(source / "portrait.jpg", exif=exif)
    Image.new("RGB", (1200, 300), "blue").save(nested / "wide.png")
    Image.new("RGB", (100, 200), "green").save(source / "small.webp")
    (source / "image_metadata.csv").write_text("filename\nportrait.jpg\n")
    np.savez(source / "vectors.npz", vectors=np.ones((1, 2)))
    (source / "index.html").write_text("<title>Book Wall</title>")
    (source / "notes.txt").write_text("do not publish")

    create_thumbnails(source, destination, verbose=0)

    with Image.open(destination / "portrait.jpg") as output:
        assert output.size == (300, 600)
        assert output.format == "JPEG"
        assert output.getexif().get(315) is None
    with Image.open(destination / "nested" / "wide.png") as output:
        assert output.size == (800, 200)
        assert output.format == "PNG"
    with Image.open(destination / "small.webp") as output:
        assert output.size == (100, 200)
        assert output.format == "WEBP"

    assert (destination / "image_metadata.csv").read_bytes() == (
        source / "image_metadata.csv"
    ).read_bytes()
    assert (destination / "vectors.npz").read_bytes() == (
        source / "vectors.npz"
    ).read_bytes()
    assert (destination / "index.html").read_bytes() == (
        source / "index.html"
    ).read_bytes()
    assert not (destination / "notes.txt").exists()


def test_create_thumbnails_preserves_source_report_status(tmp_path: Path) -> None:
    """Translate metadata and reviewed cache state to generated thumbnails."""
    source = tmp_path / "books"
    destination = tmp_path / "book_thumbs"
    source.mkdir()
    destination.mkdir()
    removed_from_source = destination / "removed_from_source.jpg"
    Image.new("RGB", (10, 10), "blue").save(removed_from_source)
    image_paths = [
        source / "reviewed.jpg",
        source / "unreviewed.jpg",
        source / "missing_metadata.jpg",
    ]
    for path in image_paths:
        Image.new("RGB", (800, 1600), "red").save(path)

    metadata_path = source / "image_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=[
                "status",
                "original_filepath",
                "original_filename",
                "clean_filename",
                "category",
                "genre",
                "tags",
            ],
        )
        writer.writeheader()
        for path in image_paths[:2]:
            writer.writerow(
                {
                    "status": "ok",
                    "original_filepath": path,
                    "original_filename": path.name,
                    "clean_filename": path.name,
                    "category": "books",
                    "genre": "reference",
                    "tags": "[]",
                }
            )

    source_entries = {
        str(path.resolve()): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            np.ones(2, dtype=np.float64),
        )
        for path in image_paths
    }
    reviewed_key = str(image_paths[0].resolve())
    _save_embedding_cache(
        source / DEDUPE_EMBEDDINGS_FILENAME,
        CLIP_MODEL,
        EmbeddingCache(
            entries=source_entries,
            reviewed_thresholds={reviewed_key: (0.99, 0.85)},
        ),
    )
    (source / "index.html").write_text("<title>Books</title>")

    create_thumbnails(source, destination, verbose=0)

    assert not removed_from_source.exists()
    source_report = report_images(source, metadata_path)
    destination_report = report_images(
        destination,
        destination / metadata_path.name,
    )
    assert destination_report == source_report
    destination_index = destination / "index.html"
    assert destination_index.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns
        for path in destination.iterdir()
        if path.suffix.lower() == ".jpg"
    )

    destination_cache = _load_embedding_cache(
        destination / DEDUPE_EMBEDDINGS_FILENAME,
        CLIP_MODEL,
    )
    destination_reviewed = destination / image_paths[0].name
    destination_key = str(destination_reviewed.resolve())
    assert set(destination_cache.reviewed_thresholds) == {destination_key}
    assert destination_cache.entries[destination_key][:2] == (
        destination_reviewed.stat().st_size,
        destination_reviewed.stat().st_mtime_ns,
    )


def test_thumbnail_cli_uses_defaults_and_resolves_source_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a shelf source and apply the default thumbnail dimensions."""
    source = tmp_path / "shelves" / "books"
    destination = tmp_path / "public" / "books"
    source.mkdir(parents=True)
    Image.new("RGB", (800, 1600), "red").save(source / "cover.jpg")
    stackmap = tmp_path / ".stackmap"
    stackmap.write_text(
        f"default: {source}\nbooks: {source}\nthumbs: {destination}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image-tagger",
            "thumbnail",
            "books",
            "thumbs",
            "--stackmap",
            str(stackmap),
            "-q",
        ],
    )

    cli.main()

    with Image.open(destination / "cover.jpg") as output:
        assert output.size == (300, 600)


def test_create_thumbnails_dry_run_and_nested_destination_guard(
    tmp_path: Path,
) -> None:
    """Avoid writes during previews and reject recursive destinations."""
    source = tmp_path / "books"
    destination = tmp_path / "thumbs"
    source.mkdir()
    Image.new("RGB", (800, 1600), "red").save(source / "cover.jpg")

    create_thumbnails(source, destination, dry_run=True, verbose=0)

    assert not destination.exists()
    with pytest.raises(ValueError, match="must not be inside"):
        create_thumbnails(source, source / "thumbs", verbose=0)


def test_create_thumbnails_compacts_default_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show dots by default and file operations at higher verbosity."""
    source = tmp_path / "books"
    source.mkdir()
    Image.new("RGB", (800, 1600), "red").save(source / "cover.jpg")
    (source / "image_metadata.csv").write_text("filename\ncover.jpg\n")

    create_thumbnails(source, tmp_path / "compact", verbose=1)

    assert capsys.readouterr().out == ".\n1 succeeded, 1 metadata\n"

    create_thumbnails(source, tmp_path / "detailed", verbose=2)

    detailed_output = capsys.readouterr().out
    assert "thumbnailing" in detailed_output
    assert "cover.jpg" in detailed_output
    assert "copying" in detailed_output
    assert "image_metadata.csv" in detailed_output
    assert detailed_output.endswith("1 succeeded, 1 metadata\n")


def test_create_thumbnails_reports_compact_error_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the final counts before propagating an image error."""
    source = tmp_path / "books"
    source.mkdir()
    (source / "broken.jpg").write_text("not an image")

    with pytest.raises(OSError):
        create_thumbnails(source, tmp_path / "thumbs", verbose=1)

    assert capsys.readouterr().out == ".\n1 error\n"


def test_create_thumbnails_skips_newer_destination_images(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not open or replace a thumbnail newer than its source image."""
    source = tmp_path / "books"
    destination = tmp_path / "thumbs"
    source.mkdir()
    destination.mkdir()
    source_path = source / "cover.jpg"
    destination_path = destination / "cover.jpg"
    source_path.write_text("not an image")
    Image.new("RGB", (20, 30), "blue").save(destination_path)
    old_time = 1_700_000_000_000_000_000
    os.utime(source_path, ns=(old_time, old_time))
    os.utime(destination_path, ns=(old_time + 1_000_000_000, old_time + 1_000_000_000))
    original_bytes = destination_path.read_bytes()

    create_thumbnails(source, destination, verbose=1)

    assert capsys.readouterr().out == "1 skipped\n"

    create_thumbnails(source, destination, verbose=2)

    assert destination_path.read_bytes() == original_bytes
    output = capsys.readouterr().out
    assert "skipping" in output
    assert "cover.jpg" in output
    assert ".success!" not in output
    assert output.endswith("1 skipped\n")