import csv
import json
import shutil
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel

import cli
import image_tagger as it
from constants import WELCOME_EXTENSIONS
from util import make_unique, quote_display_path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


CATEGORIES: list[str] = [
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


TEST_CLEAN_FILENAMES: dict[str, str] = {
    "ai.jpg": "robot_portrait.jpg",
    "art.png": "picasso.png",
    "books.jpg": "library_book.jpg",
    "books_cover.jpg": "library_book.jpg",
    "comics.png": "garfield.png",
    "comics2.png": "garfield2.png",
    "diagrams.png": "flowchart.png",
    "horror.jpg": "haunted_house.jpg",
    "hygge.png": "cozy_room.png",
    "memes.jpg": "office_meme.jpg",
    "photography.jpg": "city_street.jpg",
    "speculative.jpg": "space_station.jpg",
    "vintage.tiff": "antique_camera.tiff",
}


TEST_CATEGORIES: dict[str, str] = {
    "ai.jpg": "ai",
    "art.png": "art",
    "books.jpg": "books",
    "books_cover.jpg": "books",
    "comics.png": "comics",
    "comics2.png": "comics",
    "diagrams.png": "diagrams",
    "horror.jpg": "horror",
    "hygge.png": "hygge",
    "memes.jpg": "memes",
    "photography.jpg": "photography",
    "speculative.jpg": "speculative",
    "vintage.tiff": "vintage",
}


PROMPTS: list[str] = []


class MockVisionModelClientAdapter(it.VisionModelClientAdapter):
    """Vision adapter with deterministic test metadata."""

    provider_name = "Mock"
    model = "mock-vision"

    def vision_task(
        self,
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> it.VisionTaskResult:
        """Return canned metadata for a prompt filename."""
        PROMPTS.append(prompt)
        filename = prompt.rsplit('Current filename: "', maxsplit=1)[1].split(
            '"', maxsplit=1
        )[0]
        category = TEST_CATEGORIES[filename]
        clean_filename = TEST_CLEAN_FILENAMES[filename]
        filename_already_makes_sense = clean_filename == filename
        content = json.dumps(
            {
                "description": f"Mock description for {filename}.",
                "category": category,
                "genre": "mock",
                "tags": [category, "mock"],
                "filename_already_makes_sense": filename_already_makes_sense,
                "filename": clean_filename,
            }
        )
        return it.VisionTaskResult(content=content, model=self.model, total_tokens=0)

    def cleanup(self) -> None:
        """No-op cleanup for tests."""
        pass


@pytest.fixture
def workflow_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create a full upload workflow workspace."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    for category in CATEGORIES:
        (tmp_path / category).mkdir()
    (tmp_path / "art" / "picasso.png").touch()

    for image_path in (REPO_ROOT / "tests" / "images").iterdir():
        shutil.copy2(image_path, uploads_dir / image_path.name)

    return {
        "root": tmp_path,
        "uploads": uploads_dir,
        "metadata": uploads_dir / "image_metadata.csv",
        "gallery": uploads_dir / "index.html",
    }


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Run the CLI and capture stdout."""

    def runner(*args: str) -> str:
        """Run one CLI command."""
        stdout = StringIO()
        monkeypatch.setattr(sys, "argv", ["cli.py", *args])
        with redirect_stdout(stdout):
            cli.main()
        return stdout.getvalue()

    return runner


@pytest.fixture
def run_tag(
    monkeypatch: pytest.MonkeyPatch,
    run_cli: Callable[..., str],
) -> Callable[..., str]:
    """Run tag with the mock vision adapter."""

    def runner(uploads_dir: Path, *args: str) -> str:
        """Run one tag command."""
        monkeypatch.setattr(
            it,
            "get_vision_model_client_adapter",
            lambda provider: MockVisionModelClientAdapter(),
        )
        return run_cli("tag", str(uploads_dir), *args)

    return runner


def test_full_cli_workflow_converts_tags_renames_galleries_and_shelves(
    workflow_workspace: dict[str, Path],
    run_cli: Callable[..., str],
    run_tag: Callable[..., str],
) -> None:
    """Exercise the full upload workflow."""
    uploads_dir = workflow_workspace["uploads"]
    metadata_filename = workflow_workspace["metadata"]
    gallery_filename = workflow_workspace["gallery"]

    convert_stdout = run_cli("convert", str(uploads_dir))
    assert (
        convert_stdout.splitlines()[0]
        == f"working in {quote_display_path(uploads_dir)}"
    )
    assert "converting" in convert_stdout
    assert "ai.jpeg" in convert_stdout
    assert "ai.jpg" in convert_stdout
    assert "comics.bmp" in convert_stdout
    assert "comics2.png" in convert_stdout
    assert "renaming hygge.webp to hygge.bmp ...success!" in convert_stdout
    assert "converting hygge.bmp to hygge.png ...success!" in convert_stdout
    assert "renaming speculative.bmp to speculative.webp ...success!" in convert_stdout
    assert (
        "converting speculative.webp to speculative.jpg ...success!" in convert_stdout
    )
    assert convert_stdout.endswith(".jpg: 7\n.png: 5\n.tiff: 1\n")
    assert sorted(
        path.suffix.lower() for path in uploads_dir.iterdir() if path.is_file()
    ) == [
        ".jpg",
        ".jpg",
        ".jpg",
        ".jpg",
        ".jpg",
        ".jpg",
        ".jpg",
        ".png",
        ".png",
        ".png",
        ".png",
        ".png",
        ".tiff",
    ]
    assert (uploads_dir / "ai.jpg").exists()
    assert not (uploads_dir / "ai.jpeg").exists()
    assert (uploads_dir / "comics.png").exists()
    assert (uploads_dir / "comics2.png").exists()
    assert (uploads_dir / "diagrams.png").exists()
    assert (uploads_dir / "hygge.png").exists()
    assert (uploads_dir / "speculative.jpg").exists()
    assert (uploads_dir / "vintage.tiff").exists()
    assert all(
        path.suffix.lower() in WELCOME_EXTENSIONS
        for path in uploads_dir.iterdir()
        if path.is_file()
    )

    tag_stdout = run_tag(uploads_dir, "-q")
    assert tag_stdout == "number of image files to tag: 13\n............."

    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    assert len(rows) == 13
    assert {row["status"] for row in rows} == {"ok"}
    clean_filenames = {row["original_filename"]: row["clean_filename"] for row in rows}
    assert clean_filenames == TEST_CLEAN_FILENAMES

    rename_stdout = run_cli("rename", str(uploads_dir))
    assert "renaming" in rename_stdout
    assert "success!" in rename_stdout
    assert (uploads_dir / "picasso.png").exists()
    assert (uploads_dir / "robot_portrait.jpg").exists()
    assert (uploads_dir / "library_book.jpg").exists()
    assert (uploads_dir / "library_book2.jpg").exists()
    assert (uploads_dir / "garfield.png").exists()
    assert (uploads_dir / "garfield2.png").exists()
    assert (uploads_dir / "cozy_room.png").exists()
    assert (uploads_dir / "space_station.jpg").exists()
    assert (uploads_dir / "antique_camera.tiff").exists()

    gallery_stdout = run_cli(
        "gallery",
        str(uploads_dir),
        "--output-filename",
        str(gallery_filename),
        "--no-preview",
    )
    assert gallery_stdout == ""
    assert gallery_filename.exists()
    assert "Mock description for books.jpg." in gallery_filename.read_text(
        encoding="utf-8"
    )

    shelve_stdout = run_cli("shelve", str(uploads_dir))
    assert "moving" in shelve_stdout
    assert "success!" in shelve_stdout
    assert (workflow_workspace["root"] / "ai" / "robot_portrait.jpg").exists()
    assert (workflow_workspace["root"] / "art" / "picasso.png").exists()
    assert (workflow_workspace["root"] / "art" / "picasso2.png").exists()
    assert (workflow_workspace["root"] / "books" / "library_book.jpg").exists()
    assert (workflow_workspace["root"] / "books" / "library_book2.jpg").exists()
    assert (workflow_workspace["root"] / "comics" / "garfield.png").exists()
    assert (workflow_workspace["root"] / "comics" / "garfield2.png").exists()
    assert (workflow_workspace["root"] / "hygge" / "cozy_room.png").exists()
    assert (workflow_workspace["root"] / "speculative" / "space_station.jpg").exists()
    assert (workflow_workspace["root"] / "vintage" / "antique_camera.tiff").exists()
    assert not (uploads_dir / "picasso.png").exists()


def test_generate_gallery_creates_expected_html(tmp_path: Path) -> None:
    """Render expected gallery HTML from metadata."""
    metadata_filename = tmp_path / "image_metadata.csv"
    gallery_filename = tmp_path / "index.html"
    (tmp_path / "books_books.jpg").touch()
    (tmp_path / "fallback_original.jpg").touch()
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=[*it.csv_columns, "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(tmp_path / "books.jpg"),
                "original_filename": "books.jpg",
                "width": "20",
                "height": "20",
                "category": "books",
                "genre": "mock",
                "filename": "books_books.jpg",
                "clean_filename": "books_books.jpg",
                "filename_already_makes_sense": "False",
                "tags": "books;mock;library",
                "description": "Mock description for books.jpg.",
                "notes": "Keep this one.",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:22:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(tmp_path / "fallback_original.jpg"),
                "original_filename": "fallback_original.jpg",
                "width": "20",
                "height": "20",
                "category": "books",
                "genre": "mock",
                "filename": "missing_clean.jpg",
                "clean_filename": "missing_clean.jpg",
                "filename_already_makes_sense": "False",
                "tags": "books;mock;fallback",
                "description": "Mock description for fallback_original.jpg.",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:23:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(tmp_path / "missing_original.jpg"),
                "original_filename": "missing_original.jpg",
                "width": "20",
                "height": "20",
                "category": "books",
                "genre": "mock",
                "filename": "missing_clean.jpg",
                "clean_filename": "missing_clean.jpg",
                "filename_already_makes_sense": "False",
                "tags": "books;mock;missing",
                "description": "This missing file should not render.",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:21:57.966360",
                "status": "error",
                "original_filename": "error.jpg",
                "clean_filename": "error.jpg",
                "description": "This row should not render.",
            }
        )

    it.generate_gallery(metadata_filename, gallery_filename)

    html = gallery_filename.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Image Gallery</title>" in html
    assert "Mock (mock-vision) Image Annotation" in html
    assert 'id="searchInput"' in html
    assert html.count('class="gallery-image row mb-4"') == 2
    assert '<img src="books_books.jpg" alt="Image" class="img-fluid">' in html
    assert '<img src="fallback_original.jpg" alt="Image" class="img-fluid">' in html
    assert "06/15/26 05:20 PM" in html
    assert "<strong>Category:</strong> books" in html
    assert "<strong>Genre:</strong> mock" in html
    assert '<li class="tag-pill">library</li>' in html
    assert '<li class="tag-pill">fallback</li>' in html
    assert "Mock description for books.jpg." in html
    assert "Mock description for fallback_original.jpg." in html
    assert "Keep this one." in html
    assert "This missing file should not render." not in html
    assert "This row should not render." not in html


def test_find_images_recurses_into_subdirectories(tmp_path: Path) -> None:
    """Find supported image files below nested directories."""
    root_image = tmp_path / "root.jpg"
    nested_dir = tmp_path / "art" / "paintings"
    nested_dir.mkdir(parents=True)
    nested_image = nested_dir / "nested.png"
    ignored_text = nested_dir / "notes.txt"

    root_image.touch()
    nested_image.touch()
    ignored_text.touch()

    assert it.find_images(tmp_path) == [root_image, nested_image]


def test_make_unique_returns_original_when_available(
    tmp_path: Path,
) -> None:
    """Return unchanged paths when no collision exists."""
    path = tmp_path / "image.jpg"

    assert make_unique(path) == str(path)


@pytest.mark.parametrize(
    ("filename", "existing_filenames", "expected_filename"),
    [
        ("image.jpg", ["image.jpg"], "image2.jpg"),
        ("image.jpg", ["image.jpg", "image2.jpg"], "image3.jpg"),
        ("image1.jpg", ["image1.jpg"], "image1_2.jpg"),
        ("image1.jpg", ["image1.jpg", "image1_2.jpg"], "image1_3.jpg"),
    ],
)
def test_make_unique_uses_suffixes_two_through_nine(
    tmp_path: Path,
    filename: str,
    existing_filenames: list[str],
    expected_filename: str,
) -> None:
    """Append suffixes for filename collisions."""
    for existing_filename in existing_filenames:
        (tmp_path / existing_filename).touch()

    assert make_unique(tmp_path / filename) == str(tmp_path / expected_filename)


def test_make_unique_raises_after_suffix_nine(
    tmp_path: Path,
) -> None:
    """Raise when all supported suffixes are taken."""
    (tmp_path / "image.jpg").touch()
    for suffix in range(2, 10):
        (tmp_path / f"image{suffix}.jpg").touch()

    with pytest.raises(FileExistsError):
        make_unique(tmp_path / "image.jpg")


def test_rename_verbosity_one_prints_working_folder_and_relative_quoted_paths(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Show relative rename paths at default verbosity."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    source = uploads_dir / "image 234.jpg"
    source.touch()
    metadata_filename = uploads_dir / "image_metadata.csv"
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "original_filepath": str(source),
                "original_filename": source.name,
                "clean_filename": "handwritten_note.jpg",
            }
        )

    output = run_cli("rename", str(uploads_dir))

    assert output.splitlines()[0] == f"working in {quote_display_path(uploads_dir)}"
    assert 'renaming "image 234.jpg" to handwritten_note.jpg ...success!' in output


def test_shelve_verbosity_one_prints_parent_folder_and_relative_quoted_paths(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Show parent-relative shelve paths at default verbosity."""
    uploads_dir = tmp_path / "uploads"
    diagrams_dir = tmp_path / "diagrams"
    uploads_dir.mkdir()
    diagrams_dir.mkdir()
    source = uploads_dir / "handwritten_note.jpg"
    source.touch()
    metadata_filename = uploads_dir / "image_metadata.csv"
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "category": "diagrams",
                "original_filepath": str(source),
                "original_filename": source.name,
                "clean_filename": source.name,
            }
        )

    output = run_cli("shelve", str(uploads_dir))

    assert output.splitlines()[0] == f"working in {quote_display_path(tmp_path)}"
    assert (
        "moving uploads/handwritten_note.jpg to diagrams/handwritten_note.jpg ...success!"
        in output
    )


def test_rename_verbosity_two_prints_full_quoted_paths(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Show full rename paths at higher verbosity."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    source = uploads_dir / "My Mother's Photo.jpg"
    source.touch()
    metadata_filename = uploads_dir / "image_metadata.csv"
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "original_filepath": str(source),
                "original_filename": source.name,
                "clean_filename": "family_photo.jpg",
            }
        )

    output = run_cli("rename", str(uploads_dir), "-v")

    assert output.startswith(
        f"renaming {quote_display_path(source)} to {quote_display_path(uploads_dir / 'family_photo.jpg')} ...success!"
    )


def test_convert_verbosity_one_prints_working_folder_and_relative_paths(
    tmp_path: Path,
    run_cli: Callable[..., str],
    workflow_workspace: dict[str, Path],
) -> None:
    """Show relative convert paths at default verbosity."""
    uploads_dir = tmp_path / "convert_uploads"
    shutil.copytree(workflow_workspace["uploads"], uploads_dir)

    output = run_cli("convert", str(uploads_dir))

    assert output.splitlines()[0] == f"working in {quote_display_path(uploads_dir)}"
    assert "converting comics.bmp to comics.png ...success!" in output
    assert "renaming ai.jpeg to ai.jpg ...success!" in output
    assert str(uploads_dir) not in output.splitlines()[1]


def test_convert_verbosity_two_prints_full_quoted_paths(
    tmp_path: Path,
    run_cli: Callable[..., str],
    workflow_workspace: dict[str, Path],
) -> None:
    """Show full convert paths at higher verbosity."""
    uploads_dir = tmp_path / "convert uploads"
    shutil.copytree(workflow_workspace["uploads"], uploads_dir)
    source = uploads_dir / "comics.bmp"
    target = uploads_dir / "comics.png"

    output = run_cli("convert", str(uploads_dir), "-v")

    assert output.startswith(
        f"converting {quote_display_path(source)} to {quote_display_path(target)} ...success!"
    )


def test_convert_fixes_mismatched_image_formats(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Fix mismatched extensions before converting images."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    shutil.copy2(
        REPO_ROOT / "tests" / "images" / "hygge.webp", uploads_dir / "hygge.webp"
    )
    shutil.copy2(
        REPO_ROOT / "tests" / "images" / "speculative.bmp",
        uploads_dir / "speculative.bmp",
    )

    output = run_cli("convert", str(uploads_dir))

    assert "renaming hygge.webp to hygge.bmp ...success!" in output
    assert "converting hygge.bmp to hygge.png ...success!" in output
    assert "renaming speculative.bmp to speculative.webp ...success!" in output
    assert "converting speculative.webp to speculative.jpg ...success!" in output
    assert (uploads_dir / "hygge.png").exists()
    assert (uploads_dir / "speculative.jpg").exists()


@pytest.mark.parametrize(
    ("verbosity_args", "expected"),
    [
        (("-q",), "number of image files to tag: 13\n............."),
        ((), "books.jpg -> library_book.jpg"),
        (("-v",), "'original_filename': 'books.jpg'"),
    ],
)
def test_tag_verbosity_zero_one_and_two(
    tmp_path: Path,
    run_cli: Callable[..., str],
    run_tag: Callable[..., str],
    workflow_workspace: dict[str, Path],
    verbosity_args: tuple[str, ...],
    expected: str,
) -> None:
    """Respect tag verbosity levels."""
    uploads_dir = tmp_path / "verbosity_uploads"
    shutil.copytree(workflow_workspace["uploads"], uploads_dir)
    run_cli("convert", str(uploads_dir))

    output = run_tag(uploads_dir, *verbosity_args)

    assert expected in output


def test_tag_allows_instructions_filename_override(
    tmp_path: Path,
    run_cli: Callable[..., str],
    run_tag: Callable[..., str],
    workflow_workspace: dict[str, Path],
) -> None:
    """Use custom tagging instructions."""
    uploads_dir = tmp_path / "instructions_uploads"
    shutil.copytree(workflow_workspace["uploads"], uploads_dir)
    run_cli("convert", str(uploads_dir))
    instructions_filename = tmp_path / "instructions.md"
    instructions_filename.write_text(
        'Custom tagging instructions.\n\nCurrent filename: "{filename}"\n',
        encoding="utf-8",
    )
    PROMPTS.clear()

    run_tag(uploads_dir, "-q", "--instructions-filename", str(instructions_filename))

    assert PROMPTS
    assert all(prompt.startswith("Custom tagging instructions.") for prompt in PROMPTS)
