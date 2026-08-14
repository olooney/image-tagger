import base64
import csv
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel
from send2trash.exceptions import TrashPermissionError

import cli
import image_tagger as it
import review_app
from constants import WELCOME_EXTENSIONS
from stackmap import StackMap, find_stackmap
from util import display_path, make_unique, quote_display_path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
TEST_STACKMAP_TEMPLATE: Path = REPO_ROOT / "tests" / ".stackmap"


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
    "a.b.jpg": "dotted_photo.jpg",
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
    "a.b.jpg": "photography",
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


def wall_image_srcs(html: str) -> list[str]:
    """Return rendered wall image sources in order."""
    return re.findall(r'<img src="([^"]+)"', html)


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
        return it.VisionTaskResult(
            data=response_format.model_validate_json(content),
            model=self.model,
            total_tokens=0,
        )

    def cleanup(self) -> None:
        """No-op cleanup for tests."""
        pass


class MockSameImageClientAdapter(it.VisionModelClientAdapter):
    """Vision adapter with deterministic same-image judgements."""

    provider_name = "Mock"
    model = "mock-vision"

    def __init__(self, keep: str) -> None:
        """Store the judgement to return."""
        self.keep = keep
        self.calls: list[list[str] | str] = []
        self.prompts: list[str] = []

    def vision_task(
        self,
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> it.VisionTaskResult:
        """Return a canned same-image judgement."""
        self.calls.append(image_base64)
        self.prompts.append(prompt)
        content = json.dumps(
            {
                "thinking": f"keep {self.keep}",
                "keep": self.keep,
            }
        )
        return it.VisionTaskResult(
            data=response_format.model_validate_json(content),
            model=self.model,
            total_tokens=0,
        )

    def cleanup(self) -> None:
        """No-op cleanup for tests."""
        pass


class FakeImageComparisonMethod:
    """Comparison method returning prebuilt similarities."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        """Store scores by path name pair."""
        self.scores = scores
        self.calls: list[tuple[list[Path], list[Path] | None, int, int]] = []

    def compare(
        self,
        left_images: list[Path],
        right_images: list[Path] | None = None,
        *,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> list[it.ImageSimilarity]:
        """Return configured scores for self or rectangular comparison."""
        self.calls.append((left_images, right_images, batch_size, verbose))
        pairs: list[tuple[Path, Path]] = []
        if right_images is None:
            for left_index, left_path in enumerate(left_images):
                for right_path in left_images[left_index + 1 :]:
                    pairs.append((left_path, right_path))
        else:
            for left_path in left_images:
                for right_path in right_images:
                    if left_path != right_path:
                        pairs.append((left_path, right_path))
        return [
            it.ImageSimilarity(
                score=score,
                left_path=left_path,
                right_path=right_path,
            )
            for left_path, right_path in pairs
            if (score := self.scores.get((left_path.name, right_path.name))) is not None
        ]


@pytest.fixture(autouse=True)
def disable_browser_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent CLI workflow tests from opening browser windows."""
    monkeypatch.setattr(cli, "preview", lambda _: None)


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


def write_test_stackmap(directory: Path) -> StackMap:
    """Install the default test stack map unless a workspace-specific one already exists."""
    stackmap_filename = directory.parent / ".stackmap"
    if not stackmap_filename.exists():
        shutil.copy2(TEST_STACKMAP_TEMPLATE, stackmap_filename)
    return StackMap.load(stackmap_filename)


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Run the CLI and capture stdout."""

    def runner(*args: str) -> str:
        """Run one CLI command."""
        directory = Path(args[1])
        stackmap = write_test_stackmap(directory)
        stdout = StringIO()
        monkeypatch.setattr(
            sys,
            "argv",
            ["cli.py", args[0], str(directory), "--stackmap", str(stackmap.filename), *args[2:]],
        )
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


def test_stackmap_resolves_relative_shelves_and_discovers_parent(
    tmp_path: Path,
) -> None:
    """Resolve shelf paths from the map and find parent configurations."""
    workspace = tmp_path / "library"
    nested_directory = workspace / "work" / "images"
    nested_directory.mkdir(parents=True)
    stackmap_filename = workspace / ".stackmap"
    stackmap_filename.write_text(
        "default: shelves/inbox\nart: shelves/art # paintings and drawings\n",
        encoding="utf-8",
    )

    stackmap = StackMap.load(stackmap_filename)

    assert stackmap.default_directory == workspace / "shelves" / "inbox"
    assert stackmap.categories == ["art"]
    assert stackmap.category_descriptions == {"art": "paintings and drawings"}
    assert stackmap.directory_for("art") == workspace / "shelves" / "art"
    assert find_stackmap(nested_directory) == stackmap_filename


def test_image_tag_response_schema_allows_only_configured_categories() -> None:
    """Constrain vision task categories to the configured shelves."""
    schema = it.image_tag_response_model(["art", "books"]).model_json_schema()

    assert schema["properties"]["category"]["enum"] == ["art", "books"]


def test_tag_prompt_includes_category_descriptions(
    workflow_workspace: dict[str, Path],
) -> None:
    """Include StackMap category guidance in the model prompt."""
    PROMPTS.clear()
    image_path = workflow_workspace["uploads"] / "ai.jpg"
    Image.new("RGB", (4, 4)).save(image_path)

    it.tag_image(
        image_path,
        MockVisionModelClientAdapter(),
        categories=["ai", "art"],
        category_descriptions={"ai": "artificial intelligence and machine learning"},
    )

    assert '"ai": artificial intelligence and machine learning' in PROMPTS[-1]
    assert '"art"' in PROMPTS[-1]


def test_cli_directory_alias_uses_stackmap_before_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a configured shelf alias before a same-named local directory."""
    local_directory = tmp_path / "books"
    shelf_directory = tmp_path / "shelves" / "books"
    local_directory.mkdir()
    shelf_directory.mkdir(parents=True)
    stackmap_filename = tmp_path / ".stackmap"
    stackmap_filename.write_text(
        f"default: {local_directory}\nbooks: {shelf_directory}\n",
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_dedupe(directory: Path, **kwargs: object) -> list[it.ImageDuplicateMatch]:
        calls.append(directory)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(it, "dedupe_images", fake_dedupe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli.py", "dedupe", "books", "--stackmap", str(stackmap_filename), "-q"],
    )

    cli.main()

    local_fallback_directory = tmp_path / "ghosts"
    local_fallback_directory.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli.py", "dedupe", "ghosts", "--stackmap", str(stackmap_filename), "-q"],
    )

    cli.main()

    assert calls == [shelf_directory, Path("ghosts")]


def test_full_cli_workflow_converts_tags_renames_galleries_and_shelves(
    workflow_workspace: dict[str, Path],
    run_cli: Callable[..., str],
    run_tag: Callable[..., str],
) -> None:
    """Exercise the full upload workflow."""
    uploads_dir = workflow_workspace["uploads"]
    metadata_filename = workflow_workspace["metadata"]
    gallery_filename = workflow_workspace["gallery"]
    metadata_backup_filename = metadata_filename.with_suffix(
        f"{metadata_filename.suffix}.bak"
    )

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
    assert "converting a.b.webp to a.b.jpg ...success!" in convert_stdout
    assert "removing duplicate a.b.webp to a.b.jpg ...success!" in convert_stdout
    assert "renaming hygge.webp to hygge.bmp ...success!" in convert_stdout
    assert "converting hygge.bmp to hygge.png ...success!" in convert_stdout
    assert "renaming speculative.bmp to speculative.webp ...success!" in convert_stdout
    assert (
        "converting speculative.webp to speculative.jpg ...success!" in convert_stdout
    )
    assert convert_stdout.endswith(".jpg: 8\n.png: 5\n.tiff: 1\n")
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
        ".jpg",
        ".png",
        ".png",
        ".png",
        ".png",
        ".png",
        ".tiff",
    ]
    assert (uploads_dir / "a.b.jpg").exists()
    assert not (uploads_dir / "a.b.webp").exists()
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
    assert tag_stdout == "number of image files to tag: 14\n.............."

    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    assert len(rows) == 14
    assert {row["status"] for row in rows} == {"ok"}
    clean_filenames = {row["original_filename"]: row["clean_filename"] for row in rows}
    assert clean_filenames == TEST_CLEAN_FILENAMES
    metadata_backup_filename.touch()

    rename_stdout = run_cli("rename", str(uploads_dir))
    assert "renaming" in rename_stdout
    assert "success!" in rename_stdout
    assert (uploads_dir / "dotted_photo.jpg").exists()
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
    assert (workflow_workspace["root"] / "photography" / "dotted_photo.jpg").exists()
    assert (workflow_workspace["root"] / "speculative" / "space_station.jpg").exists()
    assert (workflow_workspace["root"] / "vintage" / "antique_camera.tiff").exists()
    assert not (uploads_dir / "picasso.png").exists()

    dedupe_review_filename = uploads_dir / it.DEDUPE_REVIEW_FILENAME
    dedupe_review_filename.write_text("review", encoding="utf-8")
    clean_stdout = run_cli("clean", str(uploads_dir))
    assert f"Removed {metadata_filename}" in clean_stdout
    assert f"Removed {metadata_backup_filename}" in clean_stdout
    assert f"Removed {gallery_filename}" in clean_stdout
    assert f"Removed {dedupe_review_filename}" in clean_stdout
    assert not metadata_filename.exists()
    assert not metadata_backup_filename.exists()
    assert not gallery_filename.exists()
    assert not dedupe_review_filename.exists()


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


def test_review_app_updates_one_based_metadata_row(tmp_path: Path) -> None:
    """Edit one metadata row through the review app."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (tmp_path / "art").mkdir()
    (tmp_path / "memes").mkdir()
    image_filename = uploads_dir / "sample.jpg"
    Image.new("RGB", (8, 8), "red").save(image_filename)
    Image.new("RGB", (8, 8), "blue").save(uploads_dir / "better_name.jpg")
    metadata_filename = uploads_dir / "image_metadata.csv"

    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(image_filename),
                "original_filename": image_filename.name,
                "width": "8",
                "height": "8",
                "category": "art",
                "genre": "mock",
                "filename": image_filename.name,
                "clean_filename": image_filename.name,
                "filename_already_makes_sense": "True",
                "tags": "art;mock",
                "description": "Original description.",
            }
        )

    review_app.set_review_metadata(
        metadata_filename,
        write_test_stackmap(uploads_dir),
    )
    client = TestClient(review_app.app)

    response = client.get("/")
    assert response.status_code == 200
    assert "Image Metadata Review" in response.text
    assert image_filename.name in response.text
    assert "Metadata:" not in response.text
    assert "Rows:" not in response.text
    assert "Delete" in response.text
    assert 'Original Filename' in response.text
    assert 'readonly' in response.text
    assert "No images to review." in response.text
    assert '<p id="empty-review-message" class="alert alert-info" role="status" hidden>' in response.text
    assert '<div id="shelve-controls" class="mb-4 d-flex align-items-start" hidden>' in response.text

    response = client.delete("/row/1")
    assert response.status_code == 200
    assert response.text == ""
    assert not image_filename.exists()

    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        deleted_row = next(csv.DictReader(metadata_file))
    assert deleted_row["status"] == "deleted"

    response = client.get("/")
    assert response.status_code == 200
    assert "No images to review." in response.text
    assert '<p id="empty-review-message" class="alert alert-info" role="status" hidden>' not in response.text
    assert '<div id="shelve-controls" class="mb-4 d-flex align-items-start" hidden>' not in response.text

    Image.new("RGB", (8, 8), "red").save(image_filename)
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(image_filename),
                "original_filename": image_filename.name,
                "width": "8",
                "height": "8",
                "category": "art",
                "genre": "mock",
                "filename": image_filename.name,
                "clean_filename": image_filename.name,
                "filename_already_makes_sense": "True",
                "tags": "art;mock",
                "description": "Original description.",
            }
        )

    response = client.get("/")
    assert response.status_code == 200
    assert "Shelve" in response.text

    response = client.post(
        "/row/1",
        data={
            "category": "memes",
            "genre": "joke",
            "clean_filename": "better_name.jpg",
            "tags": "meme;mock",
            "description": "Updated description.",
        },
    )
    assert response.status_code == 200
    assert "Saved." in response.text
    assert "memes" in response.text
    assert "better_name2.jpg" in response.text

    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        updated_row = next(csv.DictReader(metadata_file))
    assert updated_row["category"] == "memes"
    assert updated_row["genre"] == "joke"
    assert updated_row["clean_filename"] == "better_name2.jpg"
    assert updated_row["tags"] == "meme;mock"
    assert updated_row["description"] == "Updated description."
    assert not image_filename.exists()
    assert (uploads_dir / "better_name2.jpg").exists()

    response = client.post(
        "/row/1",
        data={
            "category": "memes",
            "genre": "joke",
            "clean_filename": "final_name.jpg",
            "tags": "meme;mock",
            "description": "Updated description.",
        },
    )
    assert response.status_code == 200
    assert "final_name.jpg" in response.text

    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        updated_row = next(csv.DictReader(metadata_file))
    assert updated_row["clean_filename"] == "final_name.jpg"
    assert not (uploads_dir / "better_name2.jpg").exists()
    assert (uploads_dir / "final_name.jpg").exists()
    assert metadata_filename.with_suffix(".csv.bak").exists()

    response = client.post("/shelve")
    assert response.status_code == 200
    assert "moving uploads/final_name.jpg to memes/final_name.jpg ...success!" in response.text
    assert 'id="review-list" hx-swap-oob="innerHTML"' in response.text
    assert response.text.endswith('hx-swap-oob="innerHTML"></div>')
    assert not (uploads_dir / "final_name.jpg").exists()
    assert (tmp_path / "memes" / "final_name.jpg").exists()


def test_prune_cli_removes_rows_without_existing_images(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Drop metadata rows whose source files are no longer present."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    image_filename = uploads_dir / "kept.jpg"
    Image.new("RGB", (8, 8), "red").save(image_filename)
    metadata_filename = uploads_dir / "image_metadata.csv"

    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(image_filename),
                "original_filename": "kept.jpg",
                "width": "8",
                "height": "8",
                "category": "art",
                "genre": "mock",
                "filename": "kept.jpg",
                "clean_filename": "kept.jpg",
                "filename_already_makes_sense": "True",
                "tags": "art;mock",
                "description": "Keep me.",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(uploads_dir / "missing.jpg"),
                "original_filename": "missing.jpg",
                "width": "8",
                "height": "8",
                "category": "art",
                "genre": "mock",
                "filename": "missing.jpg",
                "clean_filename": "missing.jpg",
                "filename_already_makes_sense": "True",
                "tags": "art;mock",
                "description": "Remove me.",
            }
        )

    output = run_cli("prune", str(uploads_dir))

    assert output == "removed 1 row(s) from image_metadata.csv\n"
    with metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    assert [row["original_filename"] for row in rows] == ["kept.jpg"]
    assert "Remove me." not in metadata_filename.read_text(encoding="utf-8")


def test_shelve_cli_moves_misplaced_images_from_directory_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_cli: Callable[..., str],
) -> None:
    """Shelve matching categories from any configured alias directory."""
    monkeypatch.chdir(tmp_path)
    stackmap_filename = tmp_path / ".stackmap"
    stackmap_filename.write_text(
        "default: shelves/inbox\nart: shelves/art\nbooks: shelves/books\n",
        encoding="utf-8",
    )
    books_dir = tmp_path / "shelves" / "books"
    art_dir = tmp_path / "shelves" / "art"
    books_dir.mkdir(parents=True)
    art_dir.mkdir(parents=True)
    misplaced = books_dir / "misplaced_art.jpg"
    correct = books_dir / "books.jpg"
    Image.new("RGB", (10, 10), "red").save(misplaced)
    Image.new("RGB", (10, 10), "blue").save(correct)
    metadata_filename = books_dir / "image_metadata.csv"
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(misplaced),
                "original_filename": misplaced.name,
                "width": "10",
                "height": "10",
                "category": "art",
                "genre": "mock",
                "filename": misplaced.name,
                "clean_filename": misplaced.name,
                "filename_already_makes_sense": "True",
                "tags": "art;mock",
                "description": "Move me.",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(correct),
                "original_filename": correct.name,
                "width": "10",
                "height": "10",
                "category": "books",
                "genre": "mock",
                "filename": correct.name,
                "clean_filename": correct.name,
                "filename_already_makes_sense": "True",
                "tags": "books;mock",
                "description": "Keep me here.",
            }
        )

    run_cli("shelve", "books")

    assert not misplaced.exists()
    assert (art_dir / misplaced.name).exists()
    assert correct.exists()
    assert (books_dir / "image_metadata.csv").exists()


def test_prune_metadata_rows_removes_empty_metadata_file(
    tmp_path: Path,
) -> None:
    """Delete metadata files after they are fully pruned to empty."""
    metadata_filename = tmp_path / "image_metadata.csv"
    metadata_filename.write_text(
        "timestamp,status,original_filepath,original_filename\n",
        encoding="utf-8",
    )

    assert it.prune_metadata_rows(metadata_filename, verbose=0) == 0
    assert not metadata_filename.exists()


def test_review_metadata_cleans_backup_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the review server removes its backup when interrupted."""
    metadata_filename = tmp_path / "image_metadata.csv"
    metadata_filename.write_text(
        "timestamp,status,original_filepath,original_filename\n",
        encoding="utf-8",
    )
    backup_filename = metadata_filename.with_suffix(".csv.bak")
    backup_filename.write_text("backup", encoding="utf-8")

    monkeypatch.setattr(review_app.webbrowser, "open", lambda *args, **kwargs: None)

    import uvicorn

    def fake_run(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn, "run", fake_run)

    with pytest.raises(KeyboardInterrupt):
        review_app.review_metadata(
            metadata_filename,
            stackmap=write_test_stackmap(tmp_path),
        )

    assert not backup_filename.exists()


def test_review_cli_exits_if_metadata_is_missing(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Require metadata before starting the review app."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        run_cli("review", str(uploads_dir))

    assert exc_info.value.code == f"metadata file not found: {uploads_dir / 'image_metadata.csv'}"


def test_gallery_cli_defaults_output_to_selected_directory(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Write gallery index beside metadata in the selected directory."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    image_filename = uploads_dir / "books.jpg"
    Image.new("RGB", (20, 20)).save(image_filename)
    with (uploads_dir / "image_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-15T17:20:57.966360",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(image_filename),
                "original_filename": image_filename.name,
                "width": "20",
                "height": "20",
                "category": "books",
                "genre": "mock",
                "filename": image_filename.name,
                "clean_filename": image_filename.name,
                "filename_already_makes_sense": "True",
                "tags": "books;mock",
                "description": "Mock description for books.jpg.",
            }
        )

    run_cli("gallery", str(uploads_dir), "--no-preview")

    gallery_filename = uploads_dir / "index.html"
    assert gallery_filename.exists()
    assert "Mock description for books.jpg." in gallery_filename.read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "index.html").exists()


def test_wall_cli_generates_regular_grid_with_relative_image_paths(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Render a standalone image wall from discovered images."""
    uploads_dir = tmp_path / "books"
    nested_dir = uploads_dir / "nested"
    nested_dir.mkdir(parents=True)
    square_filename = uploads_dir / "square.jpg"
    Image.new("RGB", (100, 100)).save(square_filename)
    Image.new("RGB", (200, 100)).save(nested_dir / "wide.png")
    Image.new("RGB", (500, 100)).save(uploads_dir / "wider.jpg")
    with (uploads_dir / "image_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "original_filepath": str(square_filename),
                "original_filename": square_filename.name,
                "width": "100",
                "height": "100",
                "clean_filename": square_filename.name,
                "category": "books",
                "genre": "mock",
                "tags": "library;reference",
                "description": "Mock description for square.jpg.",
            }
        )

    output = run_cli("wall", str(uploads_dir), "--no-preview")

    wall_filename = uploads_dir / "index.html"
    html = wall_filename.read_text(encoding="utf-8")
    assert output == f"wrote {quote_display_path(wall_filename)}\n"
    assert wall_filename.exists()
    assert html.startswith("<!doctype html>")
    assert "<title>Book Wall</title>" in html
    assert "--cell-width: 200px;" in html
    assert "--cell-height: 100px;" in html
    assert "grid-template-columns: repeat(auto-fill" in html
    assert "grid-auto-rows: var(--cell-height);" in html
    assert "object-fit: cover;" in html
    assert "object-position: center top;" in html
    assert "max-height: calc(100vh - 2vmin);" in html
    assert 'class="tile double-wide"' in html
    assert 'src="square.jpg"' in html
    assert 'src="nested/wide.png"' in html
    assert 'src="wider.jpg"' in html
    assert set(wall_image_srcs(html)) == {"square.jpg", "nested/wide.png", "wider.jpg"}
    assert 'title="square.jpg (100x100)' in html
    assert 'Category: books' in html
    assert 'Tags: library, reference' in html
    assert 'Mock description for square.jpg.' in html
    assert 'title="wider.jpg"' in html
    assert str(uploads_dir) not in html
    assert 'class="search-panel"' in html
    assert "const ASCII_EQUIVALENTS" in html
    assert ".normalize('NFKD')" in html
    assert "terms.every((term) => searchText.includes(term))" in html
    assert "const closeSearch" in html
    assert "event.key.toLowerCase() === 'f'" in html
    assert "event.ctrlKey || event.metaKey" in html
    assert "lightbox.classList.add('is-open')" in html
    assert "lightbox.classList.remove('is-open')" in html
    assert "event.key === 'ArrowLeft'" in html
    assert "event.key === 'ArrowRight'" in html


def test_wall_cli_uses_exif_orientation_and_reweighted_cell_aspect_ratio(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Render camera images using their display orientation and tile span."""
    uploads_dir = tmp_path / "camera"
    uploads_dir.mkdir()
    for index in range(3):
        portrait = Image.new("RGB", (400, 300))
        exif = portrait.getexif()
        exif[274] = 6
        portrait.save(uploads_dir / f"portrait-{index}.jpg", exif=exif)
    Image.new("RGB", (400, 300)).save(uploads_dir / "landscape.jpg")

    run_cli("wall", str(uploads_dir), "--no-preview")

    html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    assert "--cell-height: 267px;" in html
    assert html.count('class="tile double-wide"') == 1


def test_infer_wall_layout_reweights_double_wide_images() -> None:
    """Re-estimate the base cell ratio after assigning double-wide tiles."""
    aspect_ratios = {
        Path("narrow.jpg"): 0.9,
        Path("square.jpg"): 1.0,
        Path("base.jpg"): 1.2,
        Path("wide.jpg"): 2.1,
        Path("wider.jpg"): 2.2,
    }

    cell_aspect_ratio, double_wide_paths = it.infer_wall_layout(aspect_ratios)

    assert cell_aspect_ratio == pytest.approx(1.05)
    assert double_wide_paths == {Path("wide.jpg"), Path("wider.jpg")}


def test_wall_cli_title_can_be_overridden(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Use an explicit HTML title when provided."""
    uploads_dir = tmp_path / "books"
    uploads_dir.mkdir()
    Image.new("RGB", (100, 100)).save(uploads_dir / "square.jpg")

    run_cli("wall", str(uploads_dir), "--title", "Library Shelf", "--no-preview")

    html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    assert "<title>Library Shelf</title>" in html


def test_wall_cli_orders_images_by_date_newest_first(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Render the image wall with newest images first."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    older_filename = uploads_dir / "z-older.jpg"
    newer_filename = uploads_dir / "a-newer.jpg"
    Image.new("RGB", (100, 100)).save(older_filename)
    Image.new("RGB", (100, 100)).save(newer_filename)
    os.utime(older_filename, (100, 100))
    os.utime(newer_filename, (200, 200))

    run_cli("wall", str(uploads_dir), "--order", "date", "--no-preview")

    html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    assert html.index('src="a-newer.jpg"') < html.index('src="z-older.jpg"')


def test_wall_cli_random_order_accepts_seed(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Render deterministic random wall ordering from a seed."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    for filename in ["alpha.jpg", "bravo.jpg", "charlie.jpg", "delta.jpg"]:
        Image.new("RGB", (100, 100)).save(uploads_dir / filename)

    run_cli("wall", str(uploads_dir), "--order", "random", "--seed", "1", "--no-preview")
    first_html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    run_cli("wall", str(uploads_dir), "--order", "random", "--seed", "1", "--no-preview")
    repeated_html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    run_cli("wall", str(uploads_dir), "--order", "random", "--seed", "2", "--no-preview")
    different_html = (uploads_dir / "index.html").read_text(encoding="utf-8")

    assert first_html == repeated_html
    assert first_html != different_html


def test_wall_cli_random_order_defaults_to_seed_37(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Use seed 37 for random wall ordering when no seed is provided."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    for filename in ["alpha.jpg", "bravo.jpg", "charlie.jpg", "delta.jpg"]:
        Image.new("RGB", (100, 100)).save(uploads_dir / filename)

    run_cli("wall", str(uploads_dir), "--no-preview")
    default_seed_html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    run_cli("wall", str(uploads_dir), "--order", "random", "--seed", "37", "--no-preview")
    explicit_seed_html = (uploads_dir / "index.html").read_text(encoding="utf-8")

    assert default_seed_html == explicit_seed_html


def test_wall_cli_seeded_random_order_preserves_existing_relative_order(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Keep seeded random order stable when new images are added."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    original_filenames = ["alpha.jpg", "bravo.jpg", "charlie.jpg", "delta.jpg"]
    for filename in original_filenames:
        Image.new("RGB", (100, 100)).save(uploads_dir / filename)

    run_cli("wall", str(uploads_dir), "--order", "random", "--seed", "42", "--no-preview")
    first_html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    original_order = wall_image_srcs(first_html)

    Image.new("RGB", (100, 100)).save(uploads_dir / "echo.jpg")
    run_cli("wall", str(uploads_dir), "--order", "random", "--seed", "42", "--no-preview")
    expanded_html = (uploads_dir / "index.html").read_text(encoding="utf-8")
    expanded_original_order = [
        src for src in wall_image_srcs(expanded_html) if src in original_filenames
    ]

    assert expanded_original_order == original_order
    assert "echo.jpg" in wall_image_srcs(expanded_html)


def test_paths_with_mtime_accepts_iterators(tmp_path: Path) -> None:
    """Pair iterable file paths with modification times."""
    first_filename = tmp_path / "first.jpg"
    second_filename = tmp_path / "second.jpg"
    first_filename.touch()
    second_filename.touch()
    os.utime(first_filename, (300, 300))
    os.utime(second_filename, (400, 400))

    pairs = it.paths_with_mtime(iter([first_filename, second_filename]))

    assert pairs == [(300, first_filename), (400, second_filename)]


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


def test_display_path_uses_absolute_path_outside_relative_root(tmp_path: Path) -> None:
    """Display configured shelves outside the workspace without failing."""
    workspace = tmp_path / "workspace"
    shelf = tmp_path / "shelf" / "image.jpg"
    workspace.mkdir()
    shelf.parent.mkdir()

    assert display_path(shelf, verbose=1, relative_to=workspace) == str(shelf)


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


def test_shelve_verbosity_one_handles_shelves_outside_stackmap_directory(
    tmp_path: Path,
) -> None:
    """Move between independently located configured shelves."""
    config_directory = tmp_path / "project"
    uploads_dir = tmp_path / "uploads"
    diagrams_dir = tmp_path / "diagrams"
    config_directory.mkdir()
    uploads_dir.mkdir()
    diagrams_dir.mkdir()
    stackmap_filename = config_directory / ".stackmap"
    stackmap_filename.write_text(
        f"default: {uploads_dir}\ndiagrams: {diagrams_dir}\n",
        encoding="utf-8",
    )
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

    stdout = StringIO()
    with redirect_stdout(stdout):
        it.shelve_images(metadata_filename, stackmap=StackMap.load(stackmap_filename))

    assert f"moving {source} to {diagrams_dir / source.name} ...success!" in stdout.getvalue()
    assert (diagrams_dir / source.name).is_file()


def test_shelve_appends_metadata_to_target_directory_after_unique_filename(
    tmp_path: Path,
    run_cli: Callable[..., str],
) -> None:
    """Copy shelved metadata to target metadata with the final filename."""
    uploads_dir = tmp_path / "uploads"
    books_dir = tmp_path / "books"
    uploads_dir.mkdir()
    books_dir.mkdir()
    source = uploads_dir / "library_book.jpg"
    source.touch()
    (books_dir / "library_book.jpg").touch()

    metadata_filename = uploads_dir / "image_metadata.csv"
    with metadata_filename.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-26T12:00:00",
                "status": "ok",
                "total_tokens": "42",
                "provider_name": "Mock",
                "model": "mock-vision",
                "original_filepath": str(source),
                "original_filename": source.name,
                "width": "100",
                "height": "100",
                "category": "books",
                "genre": "mock",
                "filename": source.name,
                "clean_filename": source.name,
                "filename_already_makes_sense": "True",
                "tags": "books;mock",
                "description": "Mock description.",
            }
        )

    target_metadata_filename = books_dir / "image_metadata.csv"
    with target_metadata_filename.open(
        "w", newline="", encoding="utf-8"
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=it.csv_columns)
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "original_filepath": str(books_dir / "library_book.jpg"),
                "original_filename": "library_book.jpg",
                "clean_filename": "library_book.jpg",
            }
        )

    run_cli("shelve", str(uploads_dir))

    assert (books_dir / "library_book2.jpg").exists()
    with target_metadata_filename.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    assert len(rows) == 2
    appended_row = rows[1]
    assert appended_row["original_filepath"] == str(books_dir / "library_book2.jpg")
    assert appended_row["original_filename"] == "library_book2.jpg"
    assert appended_row["clean_filename"] == "library_book2.jpg"
    assert appended_row["description"] == "Mock description."


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
    source = uploads_dir / "a.b.webp"
    target = uploads_dir / "a.b.jpg"

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
        (("-q",), "number of image files to tag: 14\n.............."),
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


def test_dedupe_image_matches_accepts_self_comparison_upper_triangle(
    tmp_path: Path,
) -> None:
    """Accept high-score self-comparison matches from the strategy."""
    paths = [tmp_path / "b.jpg", tmp_path / "a.jpg", tmp_path / "c.jpg"]
    for path in paths:
        Image.new("RGB", (4, 4)).save(path)
    comparison_method = FakeImageComparisonMethod(
        {
            ("a.jpg", "b.jpg"): 0.995,
            ("a.jpg", "c.jpg"): 0.2,
            ("b.jpg", "c.jpg"): 0.89,
        }
    )

    matches = it.dedupe_image_matches(
        paths,
        automatic_threshold=0.99,
        llm_threshold=0.9,
        comparison_method=comparison_method,
        verbose=0,
    )

    assert comparison_method.calls[0][0] == [
        tmp_path / "a.jpg",
        tmp_path / "b.jpg",
        tmp_path / "c.jpg",
    ]
    assert comparison_method.calls[0][1] is None
    assert [(match.left_path.name, match.right_path.name) for match in matches] == [
        ("a.jpg", "b.jpg")
    ]
    assert matches[0].decision_source == "clip"


def test_dedupe_image_matches_accepts_asymmetric_rectangular_pairs(
    tmp_path: Path,
) -> None:
    """Accept high-score left/right matches for future asymmetric workflows."""
    left_a = tmp_path / "left-a.jpg"
    left_b = tmp_path / "left-b.jpg"
    right_a = tmp_path / "right-a.jpg"
    right_b = tmp_path / "right-b.jpg"
    for path in [left_a, left_b, right_a, right_b]:
        Image.new("RGB", (4, 4)).save(path)
    comparison_method = FakeImageComparisonMethod(
        {
            ("left-a.jpg", "right-a.jpg"): 0.991,
            ("left-a.jpg", "right-b.jpg"): 0.1,
            ("left-b.jpg", "right-a.jpg"): 0.4,
        }
    )

    matches = it.dedupe_image_matches(
        [left_b, left_a],
        [right_b, right_a],
        automatic_threshold=0.99,
        llm_threshold=0.9,
        comparison_method=comparison_method,
        verbose=0,
    )

    assert comparison_method.calls[0][1] == [right_a, right_b]
    assert [(match.left_path.name, match.right_path.name) for match in matches] == [
        ("left-a.jpg", "right-a.jpg")
    ]


def test_dedupe_image_matches_keeps_larger_image_automatically(
    tmp_path: Path,
) -> None:
    """Keep the larger image for automatic high-confidence matches."""
    small_path = tmp_path / "a-small.jpg"
    large_path = tmp_path / "z-large.jpg"
    Image.new("RGB", (4, 4)).save(small_path)
    Image.new("RGB", (8, 8)).save(large_path)
    comparison_method = FakeImageComparisonMethod(
        {(small_path.name, large_path.name): 0.995}
    )

    matches = it.dedupe_image_matches(
        [small_path, large_path],
        comparison_method=comparison_method,
        verbose=0,
    )

    assert [(match.left_path, match.right_path) for match in matches] == [
        (large_path, small_path)
    ]


def test_dedupe_image_matches_breaks_automatic_size_ties_by_filename(
    tmp_path: Path,
) -> None:
    """Keep the lexicographically first filename when image sizes tie."""
    first_path = tmp_path / "a-first.jpg"
    second_path = tmp_path / "b-second.jpg"
    Image.new("RGB", (4, 4)).save(first_path)
    Image.new("RGB", (4, 4)).save(second_path)
    comparison_method = FakeImageComparisonMethod(
        {(first_path.name, second_path.name): 0.995}
    )

    matches = it.dedupe_image_matches(
        [second_path, first_path],
        comparison_method=comparison_method,
        verbose=0,
    )

    assert [(match.left_path, match.right_path) for match in matches] == [
        (first_path, second_path)
    ]


def test_dedupe_image_matches_skips_llm_for_planned_removal(
    tmp_path: Path,
) -> None:
    """Avoid LLM adjudication for pairs containing a selected duplicate."""
    kept_path = tmp_path / "a-kept.jpg"
    duplicate_path = tmp_path / "b-duplicate.jpg"
    other_path = tmp_path / "c-other.jpg"
    Image.new("RGB", (10, 10)).save(kept_path)
    Image.new("RGB", (4, 4)).save(duplicate_path)
    Image.new("RGB", (4, 4)).save(other_path)
    comparison_method = FakeImageComparisonMethod(
        {
            (kept_path.name, duplicate_path.name): 1.0,
            (duplicate_path.name, other_path.name): 0.95,
        }
    )
    client_adapter = MockSameImageClientAdapter("left")

    matches = it.dedupe_image_matches(
        [kept_path, duplicate_path, other_path],
        comparison_method=comparison_method,
        client_adapter=client_adapter,
        verbose=0,
    )

    assert [(match.left_path, match.right_path) for match in matches] == [
        (kept_path, duplicate_path)
    ]
    assert client_adapter.calls == []


@pytest.mark.parametrize("provider", list(it.VisionModelProvider))
@pytest.mark.parametrize("keep", ["left", "right", "both"])
def test_dedupe_image_matches_uses_llm_for_borderline_candidates(
    tmp_path: Path,
    provider: it.VisionModelProvider,
    keep: str,
) -> None:
    """Confirm borderline pairs through an injected vision adapter."""
    left_path = tmp_path / "left.jpg"
    right_path = tmp_path / "right.jpg"
    Image.new("RGB", (4, 5)).save(left_path)
    Image.new("RGB", (6, 7)).save(right_path)
    comparison_method = FakeImageComparisonMethod({("left.jpg", "right.jpg"): 0.95})
    client_adapter = MockSameImageClientAdapter(keep)

    matches = it.dedupe_image_matches(
        [left_path, right_path],
        automatic_threshold=0.99,
        llm_threshold=0.9,
        provider=provider,
        comparison_method=comparison_method,
        client_adapter=client_adapter,
        verbose=0,
    )

    assert len(client_adapter.calls) == 1
    assert 'Left file: "left.jpg" (4x5)' in client_adapter.prompts[0]
    assert 'Right file: "right.jpg" (6x7)' in client_adapter.prompts[0]
    assert "tightly framed, fronto-parallel cover image" in client_adapter.prompts[0]
    assert "keystone distortion" in client_adapter.prompts[0]
    assert len(matches) == (0 if keep == "both" else 1)
    if keep != "both":
        assert matches[0].decision_source == "llm"
        assert matches[0].judgement_text == f"keep {keep}"
        expected_kept = left_path if keep == "left" else right_path
        expected_duplicate = right_path if keep == "left" else left_path
        assert (matches[0].left_path, matches[0].right_path) == (
            expected_kept,
            expected_duplicate,
        )
        assert (matches[0].presented_left_path, matches[0].presented_right_path) == (
            left_path,
            right_path,
        )
        review_entry = it._dedupe_review_entry(matches[0], action="")
        assert (review_entry.left_path, review_entry.right_path) == (
            left_path,
            right_path,
        )
        assert review_entry.duplicate_side == (
            "right" if keep == "left" else "left"
        )


def test_dedupe_image_matches_prints_verbose_llm_judgement_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show second-pass LLM details at high verbosity."""
    left_path = tmp_path / "left image.jpg"
    right_path = tmp_path / "right.jpg"
    Image.new("RGB", (4, 5)).save(left_path)
    Image.new("RGB", (6, 7)).save(right_path)
    comparison_method = FakeImageComparisonMethod(
        {(left_path.name, right_path.name): 0.95}
    )
    client_adapter = MockSameImageClientAdapter("both")
    rejected_llm_matches: list[it.ImageDuplicateMatch] = []

    matches = it.dedupe_image_matches(
        [left_path, right_path],
        automatic_threshold=0.99,
        llm_threshold=0.9,
        comparison_method=comparison_method,
        client_adapter=client_adapter,
        rejected_llm_matches=rejected_llm_matches,
        verbose=2,
    )

    output = capsys.readouterr().out
    assert matches == []
    assert rejected_llm_matches == [
        it.ImageDuplicateMatch(
            score=0.95,
            left_path=left_path,
            right_path=right_path,
            decision_source="llm",
            judgement_text="keep both",
            presented_left_path=left_path,
            presented_right_path=right_path,
        )
    ]
    assert "LLM duplicate candidate 95.00%:" in output
    assert f"left: {quote_display_path(left_path)} (4x5)" in output
    assert f"right: {quote_display_path(right_path)} (6x7)" in output
    assert "keep: both" in output
    assert "reason: keep both" in output


@pytest.mark.parametrize("dry_run", [False, True])
def test_dedupe_cli_removes_duplicate_with_relative_default_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_cli: Callable[..., str],
    dry_run: bool,
) -> None:
    """Remove duplicate files through the CLI while respecting dry-run."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    kept = uploads_dir / "kept.jpg"
    duplicate = uploads_dir / "duplicate.jpg"
    Image.new("RGB", (600, 300), "white").save(kept)
    Image.new("RGB", (300, 1200), "black").save(duplicate)

    monkeypatch.setattr(
        it,
        "dedupe_image_matches",
        lambda *args, **kwargs: [
            it.ImageDuplicateMatch(
                score=1.0,
                left_path=kept,
                right_path=duplicate,
                decision_source="clip",
            )
        ],
    )
    recycled_paths: list[Path] = []

    def fake_send2trash(path: Path) -> None:
        """Record the recycle action without changing the system recycle bin."""
        recycled_paths.append(path)
        path.unlink()

    monkeypatch.setattr(it, "send2trash", fake_send2trash)
    previewed_paths: list[Path] = []
    monkeypatch.setattr(cli, "preview", previewed_paths.append)
    dry_run_args = ("--dry-run",) if dry_run else ()

    output = run_cli("dedupe", str(uploads_dir), *dry_run_args)

    assert output.splitlines()[0] == f"working in {quote_display_path(uploads_dir)}"
    assert "removing duplicate duplicate.jpg to kept.jpg ...success!" in output
    assert duplicate.exists() is dry_run
    assert recycled_paths == ([] if dry_run else [duplicate])
    assert kept.exists()
    review_filename = uploads_dir / it.DEDUPE_REVIEW_FILENAME
    review_html = review_filename.read_text(encoding="utf-8")
    expected_action = (
        "Right image would be removed as a duplicate."
        if dry_run
        else "Right image was removed as a duplicate."
    )
    assert "data:image/png;base64," in review_html
    assert "Automatic CLIP match" in review_html
    assert expected_action in review_html
    assert previewed_paths == [review_filename]
    encoded_thumbnails = re.findall(r'data:image/png;base64,([^"\']+)', review_html)
    thumbnail_sizes = [
        Image.open(BytesIO(base64.b64decode(encoded_thumbnail))).size
        for encoded_thumbnail in encoded_thumbnails
    ]
    assert thumbnail_sizes == [(500, 250), (188, 750)]


def test_dedupe_permanently_deletes_when_recycle_bin_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to deletion when no recycle bin is available."""
    kept = tmp_path / "kept.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    kept.touch()
    duplicate.touch()
    monkeypatch.setattr(
        it,
        "dedupe_image_matches",
        lambda *args, **kwargs: [
            it.ImageDuplicateMatch(
                score=1.0,
                left_path=kept,
                right_path=duplicate,
                decision_source="clip",
            )
        ],
    )

    def unavailable_trash(path: Path) -> None:
        """Simulate a filesystem without a usable recycle bin."""
        raise TrashPermissionError(path)

    monkeypatch.setattr(it, "send2trash", unavailable_trash)

    it.dedupe_images(tmp_path, verbose=0)

    assert not duplicate.exists()
    assert kept.exists()


def test_dedupe_cli_passes_custom_thresholds_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_cli: Callable[..., str],
) -> None:
    """Pass dedupe CLI options through to the public API."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    calls: list[dict[str, object]] = []

    def fake_dedupe_images(directory: Path, **kwargs: object) -> list[it.ImageDuplicateMatch]:
        calls.append({"directory": directory, **kwargs})
        return []

    monkeypatch.setattr(it, "dedupe_images", fake_dedupe_images)

    run_cli(
        "dedupe",
        str(uploads_dir),
        "--automatic-threshold",
        "0.98",
        "--llm-threshold",
        "0.82",
        "--provider",
        "gemma",
        "--dry-run",
        "-v",
    )

    assert calls == [
        {
            "directory": uploads_dir,
            "automatic_threshold": 0.98,
            "llm_threshold": 0.82,
            "verbose": 2,
            "dry_run": True,
            "provider": "gemma",
        }
    ]


def test_dedupe_help_shows_default_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show dedupe option defaults in help output."""
    stdout = StringIO()
    monkeypatch.setattr(sys, "argv", ["cli.py", "dedupe", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        with redirect_stdout(stdout):
            cli.main()

    assert exc_info.value.code == 0
    output = stdout.getvalue()
    assert "--automatic-threshold AUTOMATIC_THRESHOLD" in output
    assert "(default: 0.99)" in output
    assert "--llm-threshold LLM_THRESHOLD" in output
    assert "(default: 0.85)" in output
    assert "--provider {openai,gemma,qwen}" in output
    assert "(default: openai)" in output
