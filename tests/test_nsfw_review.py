import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from image_tagger import cli, review_app, sfw
from image_tagger.constants import WELCOME_EXTENSIONS
from image_tagger.stackmap import StackMap


@pytest.fixture
def review_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, StackMap]:
    """Create an isolated review row without changing a stackmap file."""
    monkeypatch.setattr(review_app.app.state, "_state", {})
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "status": "ok",
                "original_filepath": str(image_path),
                "original_filename": image_path.name,
                "clean_filename": image_path.name,
                "category": "art",
                "description": "Keep this description",
            },
        ],
    ).to_csv(metadata_path, index=False)
    stackmap = StackMap(
        filename=tmp_path / "config.yaml",
        shelves={"default": tmp_path, "art": tmp_path / "art"},
        shelf_descriptions={},
    )
    return metadata_path, stackmap


@pytest.mark.parametrize("command", ["sfw", "run"])
def test_nsfw_parser_defaults(command: str) -> None:
    """Share OpenAI defaults while keeping NSFW options namespaced."""
    args = cli.build_parser().parse_args([command])

    assert args.nsfw_model == ["openai"]
    assert args.nsfw_threshold == 0.5
    assert args.nsfw_vote_threshold == 1
    assert args.nsfw_grid_depth is None
    assert args.nsfw_extensions == WELCOME_EXTENSIONS


@pytest.mark.parametrize("command", ["review", "run"])
def test_review_parser_threshold_defaults(command: str) -> None:
    """Use probability thresholds rather than percentage thresholds."""
    args = cli.build_parser().parse_args([command])

    assert args.nsfw_show_threshold == 0.1
    assert args.nsfw_highlight_threshold == 0.2


@pytest.mark.parametrize("command", ["sfw", "run"])
@pytest.mark.parametrize("model_option", ["--nsfw-model", "---nsfw-model"])
def test_nsfw_parser_custom_options(command: str, model_option: str) -> None:
    """Accept namespaced classifier options and the legacy model alias."""
    args = cli.build_parser().parse_args(
        [
            command,
            model_option,
            "openai,marqo",
            "--nsfw-threshold",
            "0.7",
            "--nsfw-vote-threshold",
            "2",
            "--nsfw-grid-depth",
            "3",
            "--nsfw-extensions",
            "png,jpg",
        ],
    )

    assert args.nsfw_model == ["openai", "marqo"]
    assert args.nsfw_threshold == 0.7
    assert args.nsfw_vote_threshold == 2
    assert args.nsfw_grid_depth == 3
    assert args.nsfw_extensions == [".png", ".jpg"]


@pytest.mark.parametrize("command", ["review", "run"])
@pytest.mark.parametrize("option", ["--nsfw-show-threshold", "--nsfw-highlight-threshold"])
@pytest.mark.parametrize("value", ["-0.01", "1.01", "20"])
def test_review_parser_rejects_out_of_range_probabilities(
    command: str,
    option: str,
    value: str,
) -> None:
    """Reject values outside the closed probability interval."""
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([command, option, value])

    assert error.value.code == 2


@pytest.mark.parametrize("command", ["review", "run"])
@pytest.mark.parametrize("value", ["0", "1"])
def test_review_parser_accepts_probability_endpoints(command: str, value: str) -> None:
    """Allow both endpoints for display and highlight thresholds."""
    args = cli.build_parser().parse_args(
        [command, "--nsfw-show-threshold", value, "--nsfw-highlight-threshold", value],
    )

    assert args.nsfw_show_threshold == float(value)
    assert args.nsfw_highlight_threshold == float(value)


@pytest.mark.parametrize(
    "option",
    [
        "--extensions",
        "--threshold",
        "--vote-threshold",
        "--grid-depth",
    ],
)
def test_sfw_parser_rejects_replaced_options(option: str) -> None:
    """Do not retain unnamespaced aliases for the sfw-only options."""
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["sfw", option, "1"])

    assert error.value.code == 2


@pytest.mark.parametrize("command", ["tag", "run"])
def test_tagging_extensions_remain_separate(command: str) -> None:
    """Preserve tagging's extension option independently of NSFW selection."""
    arguments = [command, "--extensions", "jpg"]
    if command == "run":
        arguments.extend(["--nsfw-extensions", "png"])
    args = cli.build_parser().parse_args(arguments)

    assert args.extensions == [".jpg"]
    if command == "run":
        assert args.nsfw_extensions == [".png"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_sfw_handler_forwards_metadata_and_uses_nsfw_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
) -> None:
    """Forward persistence and dry-run options without reusing tag extensions."""
    paths = [tmp_path / "image.png"]
    discovery: list[tuple[Path, list[str]]] = []
    classified: list[dict[str, Any]] = []

    def find_images(directory: Path, *, extension_filter: list[str]) -> list[Path]:
        """Record the NSFW discovery filter."""
        discovery.append((directory, extension_filter))
        return paths

    def classify_images(filepaths: list[Path], **kwargs: Any) -> sfw.SfwSummary:
        """Record forwarded options without inference."""
        classified.append({"filepaths": filepaths, **kwargs})
        return sfw.SfwSummary(total=1, sfw=1, nsfw=0, errors=0)

    monkeypatch.setattr(cli, "find_images", find_images)
    monkeypatch.setattr(sfw, "classify_images", classify_images)
    args = argparse.Namespace(
        directory=tmp_path,
        extensions=[".jpg"],
        nsfw_extensions=[".png"],
        nsfw_model=["openai", "marqo"],
        nsfw_threshold=0.7,
        nsfw_vote_threshold=2,
        nsfw_grid_depth=3,
        metadata_filename=tmp_path / "metadata.csv",
        dry_run=dry_run,
        verbose=0,
    )

    cli.sfw(args)

    assert discovery == [(tmp_path, [".png"])]
    assert classified == [
        {
            "filepaths": paths,
            "providers": ["openai", "marqo"],
            "threshold": 0.7,
            "vote_threshold": 2,
            "grid_depth": 3,
            "metadata_filename": tmp_path / "metadata.csv",
            "dry_run": dry_run,
            "verbose": 0,
        },
    ]


def test_run_places_sfw_immediately_after_quad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run safety scoring after rename and quad, but before opening review."""
    calls: list[tuple[str, argparse.Namespace]] = []
    args = argparse.Namespace()

    def record(name: str) -> Callable[[argparse.Namespace], None]:
        """Build a handler recording workflow order and shared arguments."""
        def handler(received: argparse.Namespace) -> None:
            """Record one workflow step."""
            calls.append((name, received))

        return handler

    for name in ["convert", "tag", "rename", "quad", "sfw", "review"]:
        monkeypatch.setattr(cli, name, record(name))

    cli.run(args)

    assert calls == [(name, args) for name in ["convert", "tag", "rename", "quad", "sfw", "review"]]


def test_run_forwards_nsfw_and_review_options(
    review_metadata: tuple[Path, StackMap],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real sfw and review handlers within the run workflow."""
    metadata_path, stackmap = review_metadata
    args = cli.build_parser().parse_args(
        [
            "run",
            str(metadata_path.parent),
            "--nsfw-model",
            "marqo,openai",
            "--nsfw-threshold",
            "0.6",
            "--nsfw-vote-threshold",
            "2",
            "--nsfw-grid-depth",
            "0",
            "--nsfw-extensions",
            "jpg",
            "--nsfw-show-threshold",
            "0.15",
            "--nsfw-highlight-threshold",
            "0.35",
            "--dry-run",
        ],
    )
    args.metadata_filename = metadata_path
    args.stackmap_config = stackmap
    args.verbose = 0
    classified: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    for name in ["convert", "tag", "rename", "quad"]:
        monkeypatch.setattr(cli, name, lambda args: None)

    def classify(filepaths: list[Path], **kwargs: Any) -> sfw.SfwSummary:
        """Capture classifier arguments instead of loading models."""
        classified.append(kwargs)
        return sfw.SfwSummary(total=len(filepaths), sfw=0, nsfw=0, errors=0)

    def review(metadata_filename: Path, **kwargs: Any) -> None:
        """Capture review arguments instead of starting a server."""
        reviewed.append({"metadata_filename": metadata_filename, **kwargs})

    monkeypatch.setattr(sfw, "classify_images", classify)
    monkeypatch.setattr(review_app, "review_metadata", review)

    cli.run(args)

    assert classified == [
        {
            "providers": ["marqo", "openai"],
            "threshold": 0.6,
            "vote_threshold": 2,
            "grid_depth": 0,
            "verbose": 0,
            "metadata_filename": metadata_path,
            "dry_run": True,
        },
    ]
    assert reviewed == [
        {
            "metadata_filename": metadata_path,
            "stackmap": stackmap,
            "provider": "openai",
            "verbose": 0,
            "filename_glob": None,
            "nsfw_show_threshold": 0.15,
            "nsfw_highlight_threshold": 0.35,
        },
    ]


@pytest.mark.parametrize(
    ("score", "percentage", "highlight"),
    [
        ("0", None, False),
        ("0.1", None, False),
        ("0.10001", 10, False),
        ("0.125", 13, False),
        ("0.145", 15, False),
        ("0.2", 20, False),
        ("0.20001", 20, True),
        ("0.205", 21, True),
        ("0.995", 100, True),
        ("1", 100, True),
    ],
)
def test_review_items_strict_thresholds_and_half_up_rounding(
    review_metadata: tuple[Path, StackMap],
    score: str,
    percentage: int | None,
    highlight: bool,
) -> None:
    """Compare exact probabilities before rounding percentages half up."""
    metadata_path, stackmap = review_metadata
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    metadata["nsfw_score"] = score
    metadata.to_csv(metadata_path, index=False)
    review_app.set_review_metadata(metadata_path, stackmap)

    item = review_app.review_items(metadata_path)[0]

    assert item["nsfw_percentage"] == percentage
    if percentage is not None:
        assert type(item["nsfw_percentage"]) is int
    assert item["nsfw_highlight"] is highlight


@pytest.mark.parametrize(
    "score",
    [
        None,
        "",
        " ",
        "not-a-number",
        "NaN",
        "inf",
        "-inf",
        "-0.1",
        "1.1",
    ],
)
def test_review_items_missing_or_invalid_scores_are_silent(
    review_metadata: tuple[Path, StackMap],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    score: str | None,
) -> None:
    """Keep legacy metadata and malformed scores reviewable without warnings."""
    metadata_path, stackmap = review_metadata
    if score is not None:
        metadata = pd.read_csv(metadata_path, keep_default_na=False)
        metadata["nsfw_score"] = score
        metadata.to_csv(metadata_path, index=False)
    original = metadata_path.read_bytes()
    review_app.set_review_metadata(metadata_path, stackmap)

    items = review_app.review_items(metadata_path)

    assert len(items) == 1
    assert items[0]["nsfw_percentage"] is None
    assert items[0]["nsfw_highlight"] is False
    assert items[0]["description"] == "Keep this description"
    assert metadata_path.read_bytes() == original
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert not caplog.records


@pytest.mark.parametrize(
    ("score", "show", "highlight_threshold", "percentage", "highlight"),
    [
        (0.3, 0.3, 0.4, None, False),
        (0.30001, 0.3, 0.4, 30, False),
        (0.4, 0.3, 0.4, 40, False),
        (0.40001, 0.3, 0.4, 40, True),
        (0.0, 0.0, 0.0, None, False),
        (1.0, 1.0, 1.0, None, False),
    ],
)
def test_review_items_custom_thresholds(
    review_metadata: tuple[Path, StackMap],
    score: float,
    show: float,
    highlight_threshold: float,
    percentage: int | None,
    highlight: bool,
) -> None:
    """Honor configured probability thresholds including endpoint equality."""
    metadata_path, stackmap = review_metadata
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    metadata["nsfw_score"] = score
    metadata.to_csv(metadata_path, index=False)
    review_app.set_review_metadata(
        metadata_path,
        stackmap,
        nsfw_show_threshold=show,
        nsfw_highlight_threshold=highlight_threshold,
    )

    item = review_app.review_items(metadata_path)[0]

    assert item["nsfw_percentage"] == percentage
    assert item["nsfw_highlight"] is highlight


@pytest.mark.parametrize("custom", [False, True])
def test_review_server_forwards_thresholds_to_app_state(
    review_metadata: tuple[Path, StackMap],
    monkeypatch: pytest.MonkeyPatch,
    custom: bool,
) -> None:
    """Configure thresholds before starting the review server."""
    import uvicorn

    metadata_path, stackmap = review_metadata
    show, highlight = (0.3, 0.4) if custom else (0.1, 0.2)
    started: list[tuple[float, float]] = []

    def serve(app: Any, **kwargs: Any) -> None:
        """Inspect configured state without starting a network listener."""
        started.append((app.state.nsfw_show_threshold, app.state.nsfw_highlight_threshold))

    monkeypatch.setattr(review_app, "first_available_port", lambda port: 8001)
    monkeypatch.setattr(review_app.webbrowser, "open", lambda url: None)
    monkeypatch.setattr(uvicorn, "run", serve)

    if custom:
        review_app.review_metadata(
            metadata_path,
            stackmap=stackmap,
            nsfw_show_threshold=show,
            nsfw_highlight_threshold=highlight,
        )
    else:
        review_app.review_metadata(metadata_path, stackmap=stackmap)

    assert started == [(show, highlight)]


@pytest.mark.parametrize(
    ("percentage", "highlight"),
    [(None, False), (0, False), (13, False), (20, False), (21, True)],
)
def test_review_card_renders_conditional_percentage_and_highlight(
    review_metadata: tuple[Path, StackMap],
    percentage: int | None,
    highlight: bool,
) -> None:
    """Render visible scores including zero and highlight only flagged badges."""
    metadata_path, stackmap = review_metadata
    review_app.set_review_metadata(metadata_path, stackmap)
    item = review_app.review_items(metadata_path)[0]
    item.update(nsfw_percentage=percentage, nsfw_highlight=highlight)

    rendered = review_app.card_template.render(item=item, categories=[], genres=[])

    if percentage is None:
        assert "NSFW" not in rendered
    else:
        assert f"NSFW {percentage}%" in rendered
        badge = re.search(
            rf"<(?P<tag>\w+)(?P<attributes>[^>]*)>\s*NSFW {percentage}%\s*</(?P=tag)>",
            rendered,
        )
        assert badge is not None
        assert ("filename-mismatch" in badge["attributes"]) is highlight


def test_nsfw_badge_is_right_aligned_in_review_actions() -> None:
    """Place the safety badge on the right side of the actions row."""
    rendered = review_app.page_template.render(items=[], categories=[], genres=[])
    actions = re.search(r"\.review-actions\s*\{(?P<rules>[^}]*)\}", rendered)
    badge = re.search(r"\.nsfw-score\s*\{(?P<rules>[^}]*)\}", rendered)

    assert actions is not None
    assert re.search(r"display\s*:\s*flex", actions["rules"])
    assert badge is not None
    assert re.search(r"margin-left\s*:\s*auto", badge["rules"])
