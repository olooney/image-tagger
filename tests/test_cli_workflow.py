import csv
import json
import shutil
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel

import cli
import image_tagger as it
from constants import WELCOME_EXTENSIONS


REPO_ROOT = Path(__file__).resolve().parents[1]


CATEGORIES = [
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


class MockVisionModelClientAdapter(it.VisionModelClientAdapter):
    provider_name = "Mock"
    model = "mock-vision"

    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> it.VisionTaskResult:
        filename = prompt.rsplit('Current filename: "', maxsplit=1)[1].split(
            '"', maxsplit=1
        )[0]
        stem = Path(filename).stem
        extension = Path(filename).suffix.lower()
        category = stem.split("_", maxsplit=1)[0]
        filename_already_makes_sense = filename == "art.png"
        clean_filename = (
            filename if filename_already_makes_sense else f"{stem}_{stem}{extension}"
        )
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
        pass


@pytest.fixture
def workflow_workspace(tmp_path: Path) -> dict[str, Path]:
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    for category in CATEGORIES:
        (tmp_path / category).mkdir()

    for image_path in (REPO_ROOT / "tests" / "images").iterdir():
        shutil.copy2(image_path, uploads_dir / image_path.name)

    return {
        "root": tmp_path,
        "uploads": uploads_dir,
        "metadata": uploads_dir / "image_metadata.csv",
        "gallery": uploads_dir / "index.html",
    }


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch):
    def runner(*args: str) -> str:
        stdout = StringIO()
        monkeypatch.setattr(sys, "argv", ["cli.py", *args])
        with redirect_stdout(stdout):
            cli.main()
        return stdout.getvalue()

    return runner


@pytest.fixture
def run_tag(monkeypatch: pytest.MonkeyPatch, run_cli):
    def runner(uploads_dir: Path, *args: str) -> str:
        monkeypatch.setattr(
            it,
            "get_vision_model_client_adapter",
            lambda provider: MockVisionModelClientAdapter(),
        )
        return run_cli("tag", str(uploads_dir), *args)

    return runner


def test_full_cli_workflow_converts_tags_renames_galleries_and_shelves(
    workflow_workspace: dict[str, Path], run_cli, run_tag
) -> None:
    uploads_dir = workflow_workspace["uploads"]
    metadata_filename = workflow_workspace["metadata"]
    gallery_filename = workflow_workspace["gallery"]

    convert_stdout = run_cli("convert", str(uploads_dir))
    assert "Converted" in convert_stdout
    assert convert_stdout.endswith(".gif: 1\n.jpg: 5\n.png: 1\n")
    assert sorted(
        path.suffix.lower() for path in uploads_dir.iterdir() if path.is_file()
    ) == [".gif", ".jpg", ".jpg", ".jpg", ".jpg", ".jpg", ".png"]
    assert all(
        path.suffix.lower() in WELCOME_EXTENSIONS
        for path in uploads_dir.iterdir()
        if path.is_file()
    )

    tag_stdout = run_tag(uploads_dir, "-q")
    assert tag_stdout == "number of image files to tag: 7\n......."

    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    assert len(rows) == 7
    assert {row["status"] for row in rows} == {"ok"}
    clean_filenames = {row["original_filename"]: row["clean_filename"] for row in rows}
    assert clean_filenames["art.png"] == "art.png"
    assert clean_filenames["books.jpg"] == "books_books.jpg"

    rename_stdout = run_cli("rename", str(uploads_dir))
    assert "renaming" in rename_stdout
    assert "success!" in rename_stdout
    assert (uploads_dir / "art.png").exists()
    assert (uploads_dir / "books_books.jpg").exists()
    assert (uploads_dir / "comics_comics.jpg").exists()

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
    assert (workflow_workspace["root"] / "art" / "art.png").exists()
    assert (workflow_workspace["root"] / "books" / "books_books.jpg").exists()
    assert (workflow_workspace["root"] / "comics" / "comics_comics.jpg").exists()
    assert not (uploads_dir / "art.png").exists()


def test_generate_gallery_creates_expected_html(tmp_path: Path) -> None:
    metadata_filename = tmp_path / "image_metadata.csv"
    gallery_filename = tmp_path / "index.html"
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
    assert html.count('class="gallery-image row mb-4"') == 1
    assert '<img src="books_books.jpg" alt="Image" class="img-fluid">' in html
    assert "06/15/26 05:20 PM" in html
    assert "<strong>Category:</strong> books" in html
    assert "<strong>Genre:</strong> mock" in html
    assert '<li class="tag-pill">library</li>' in html
    assert "Mock description for books.jpg." in html
    assert "Keep this one." in html
    assert "This row should not render." not in html


@pytest.mark.parametrize(
    ("verbosity_args", "expected"),
    [
        (("-q",), "number of image files to tag: 7\n......."),
        ((), "books.jpg -> books_books.jpg"),
        (("-v",), "'original_filename': 'books.jpg'"),
    ],
)
def test_tag_verbosity_zero_one_and_two(
    tmp_path: Path,
    run_cli,
    run_tag,
    workflow_workspace: dict[str, Path],
    verbosity_args: tuple[str, ...],
    expected: str,
) -> None:
    uploads_dir = tmp_path / "verbosity_uploads"
    shutil.copytree(workflow_workspace["uploads"], uploads_dir)
    run_cli("convert", str(uploads_dir))

    output = run_tag(uploads_dir, *verbosity_args)

    assert expected in output
