import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from image_tagger import cli, sfw


class RecordingClassifier:
    """Record prepared regions and return configured NSFW scores."""

    def __init__(self, scores: list[float]) -> None:
        """Store scores in inference order."""
        self.scores = iter(scores)
        self.calls: list[tuple[Image.Image, dict[str, Any]]] = []

    def classify(
        self,
        image: Image.Image,
        *,
        input_details: dict[str, Any],
        threshold: float = 0.5,
    ) -> sfw.SfwClassification:
        """Record a copy of the region and return its configured score."""
        nsfw_score = next(self.scores)
        image_copy = image.copy()
        self.calls.append((image_copy, input_details))
        scores: dict[sfw.SafetyLabel, float] = {
            "SFW": 1.0 - nsfw_score,
            "NSFW": nsfw_score,
        }
        label: sfw.SafetyLabel = "NSFW" if nsfw_score >= threshold else "SFW"
        return sfw.SfwClassification(
            label=label,
            confidence=scores[label],
            scores=scores,
            input_details=input_details,
            output_details={"scores": scores},
        )


class MockClassifier:
    """Return configured classifier results or errors."""

    def __init__(self, results: dict[str, sfw.SfwClassification | Exception]) -> None:
        """Store results by filename."""
        self.results = results

    def classify(
        self,
        image: Image.Image,
        *,
        input_details: dict[str, Any],
        threshold: float = 0.5,
    ) -> sfw.SfwClassification:
        """Return the configured result."""
        filename = Path(str(input_details["filepath"])).name
        result = self.results[filename]
        if isinstance(result, Exception):
            raise result
        return result


def classification(label: sfw.SafetyLabel) -> sfw.SfwClassification:
    """Build a deterministic test classification."""
    scores: dict[sfw.SafetyLabel, float] = (
        {"SFW": 0.1, "NSFW": 0.9} if label == "NSFW" else {"SFW": 0.8, "NSFW": 0.2}
    )
    return sfw.SfwClassification(
        label=label,
        confidence=scores[label],
        scores=scores,
        input_details={"filepath": f"images/{label.lower()}.jpg", "size": [4, 4]},
        output_details={"label": label, "scores": scores},
    )


@pytest.mark.parametrize(
    ("size", "depth"),
    [
        ((499, 200), 0),
        ((500, 200), 1),
        ((2000, 800), 1),
        ((2001, 800), 2),
    ],
)
def test_default_grid_depth_uses_longest_side(
    size: tuple[int, int],
    depth: int,
) -> None:
    """Select automatic scan depth at the configured boundaries."""
    assert sfw.default_grid_depth(size) == depth


def test_gestalt_adds_black_padding_without_cropping(tmp_path: Path) -> None:
    """Center the complete image in a square black canvas."""
    image_path = tmp_path / "landscape.png"
    Image.new("RGB", (4, 2), "white").save(image_path)
    classifier = RecordingClassifier([0.1])

    result = sfw._classify_with_tiles(
        image_path,
        classifier,
        threshold=0.5,
        grid_depth=0,
    )

    prepared, details = classifier.calls[0]
    assert result.label == "SFW"
    assert prepared.size == (4, 4)
    assert prepared.getpixel((0, 0)) == (0, 0, 0)
    assert prepared.getpixel((0, 1)) == (255, 255, 255)
    assert details["scan"] == "gestalt"


def test_overlapping_grids_cover_the_image_and_stop_by_depth(
    tmp_path: Path,
) -> None:
    """Run every tile in a grid before skipping deeper grids on detection."""
    image_path = tmp_path / "portrait.png"
    Image.new("RGB", (710, 1200), "white").save(image_path)
    classifier = RecordingClassifier([0.1, 0.2, 0.8, 0.3, 0.4, *([0.1] * 9)])

    result = sfw._classify_with_tiles(
        image_path,
        classifier,
        threshold=0.5,
        grid_depth=2,
    )

    assert result.label == "NSFW"
    assert result.scores["NSFW"] == 0.8
    assert len(classifier.calls) == 5
    tile_calls = classifier.calls[1:]
    assert [call[0].size for call in tile_calls] == [(600, 600)] * 4
    assert [call[1]["crop_box"] for call in tile_calls] == [
        [0, 0, 600, 600],
        [110, 0, 710, 600],
        [0, 600, 600, 1200],
        [110, 600, 710, 1200],
    ]


def test_grid_three_centers_middle_tiles(tmp_path: Path) -> None:
    """Center intermediate tile positions between aligned outer tiles."""
    image_path = tmp_path / "portrait.png"
    Image.new("RGB", (710, 1200), "white").save(image_path)
    classifier = RecordingClassifier([0.1] * 14)

    sfw._classify_with_tiles(
        image_path,
        classifier,
        threshold=0.5,
        grid_depth=2,
    )

    grid_three = [
        details
        for _, details in classifier.calls
        if details["grid_size"] == 3
    ]
    assert len(grid_three) == 9
    assert sorted({details["crop_box"][0] for details in grid_three}) == [0, 155, 310]
    assert sorted({details["crop_box"][1] for details in grid_three}) == [0, 400, 800]


@pytest.mark.parametrize(("grid_depth", "result_count"), [(1, 5), (2, 14)])
def test_maximum_verbosity_reports_every_scan_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    grid_depth: int,
    result_count: int,
) -> None:
    """Print the gestalt and every visited tile as separate detailed results."""
    image_path = tmp_path / "portrait.png"
    Image.new("RGB", (710, 1200), "white").save(image_path)
    classifier = RecordingClassifier([0.1] * result_count)
    monkeypatch.setattr(sfw, "load_sfw_classifier", lambda provider: classifier)

    sfw.classify_images(
        [image_path],
        grid_depth=grid_depth,
        verbose=3,
    )

    output = capsys.readouterr().out
    assert output.count("Falconsai/nsfw_image_detection: label=") == result_count
    assert output.count("Falconsai/nsfw_image_detection input:") == result_count
    assert output.count("Falconsai/nsfw_image_detection output:") == result_count


def test_default_verbosity_prints_only_nsfw_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hide SFW per-image output while always printing summary ratios."""
    sfw_path = tmp_path / "safe image.jpg"
    nsfw_path = tmp_path / "flagged.jpg"
    Image.new("RGB", (16, 16)).save(sfw_path)
    Image.new("RGB", (16, 16)).save(nsfw_path)
    classifier = MockClassifier(
        {
            sfw_path.name: classification("SFW"),
            nsfw_path.name: classification("NSFW"),
        }
    )
    monkeypatch.setattr(sfw, "load_sfw_classifier", lambda provider: classifier)

    summary = sfw.classify_images([sfw_path, nsfw_path], verbose=1)

    output = capsys.readouterr().out
    assert "loading " not in output
    assert '"' + str(sfw_path) + '"' not in output
    assert str(nsfw_path) in output
    assert "label=NSFW confidence=90.00%" in output
    assert "scores=(" not in output
    assert "NSFW: 1/2 (50.0%)" in output
    assert "\nSFW:" not in output
    assert "\nErrors:" not in output
    assert summary == sfw.SfwSummary(total=2, sfw=1, nsfw=1, errors=0)


def test_increased_and_maximum_verbosity_print_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show both labels at increased verbosity and JSON details at maximum verbosity."""
    image_path = tmp_path / "safe.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    classifier = MockClassifier({image_path.name: classification("SFW")})
    monkeypatch.setattr(sfw, "load_sfw_classifier", lambda provider: classifier)

    sfw.classify_images([image_path], verbose=2)
    verbose_output = capsys.readouterr().out
    assert "loading Falconsai/nsfw_image_detection ...success!" in verbose_output
    assert "label=SFW confidence=80.00%" in verbose_output
    assert "input:" not in verbose_output

    sfw.classify_images([image_path], verbose=3)
    maximum_output = capsys.readouterr().out
    assert "Falconsai/nsfw_image_detection: label=SFW confidence=80.00%" in maximum_output
    assert 'Falconsai/nsfw_image_detection input: {"filepath": "images/sfw.jpg", "size": [4, 4]}' in maximum_output
    assert 'Falconsai/nsfw_image_detection output: {"label": "SFW"' in maximum_output


def test_multiple_models_vote_and_report_lowest_voter_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Aggregate votes while retaining separate maximum-verbosity results."""
    image_path = tmp_path / "flagged.jpg"
    Image.new("RGB", (16, 16)).save(image_path)

    def scored_result(nsfw_score: float) -> sfw.SfwClassification:
        label: sfw.SafetyLabel = "NSFW" if nsfw_score >= 0.5 else "SFW"
        scores: dict[sfw.SafetyLabel, float] = {
            "SFW": 1.0 - nsfw_score,
            "NSFW": nsfw_score,
        }
        return sfw.SfwClassification(
            label=label,
            confidence=scores[label],
            scores=scores,
            input_details={"provider": label},
            output_details={"scores": scores},
        )

    classifiers = {
        "falconsai": MockClassifier({image_path.name: scored_result(0.9)}),
        "marqo": MockClassifier({image_path.name: scored_result(0.7)}),
        "adamcodd": MockClassifier({image_path.name: scored_result(0.2)}),
    }
    monkeypatch.setattr(
        sfw,
        "load_sfw_classifier",
        lambda provider: classifiers[provider],
    )

    sfw.classify_images(
        [image_path],
        providers=["falconsai", "marqo", "adamcodd"],
        vote_threshold=2,
        verbose=3,
    )

    output = capsys.readouterr().out
    assert "label=NSFW votes=2 confidence=70.00%" in output
    assert "Falconsai/nsfw_image_detection: label=NSFW" in output
    assert "Marqo/nsfw-image-detection-384: label=NSFW" in output
    assert "AdamCodd/vit-base-nsfw-detector: label=SFW" in output

    sfw.classify_images(
        [image_path],
        providers=["falconsai", "marqo", "adamcodd"],
        vote_threshold=3,
        verbose=2,
    )
    sfw_output = capsys.readouterr().out
    assert "label=SFW votes=2 confidence=10.00%" in sfw_output


def test_errors_are_counted_and_do_not_stop_other_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Continue after inference errors and include them in the summary."""
    broken_path = tmp_path / "broken.jpg"
    safe_path = tmp_path / "safe.jpg"
    Image.new("RGB", (16, 16)).save(broken_path)
    Image.new("RGB", (16, 16)).save(safe_path)
    classifier = MockClassifier(
        {
            broken_path.name: ValueError("bad image"),
            safe_path.name: classification("SFW"),
        }
    )
    monkeypatch.setattr(sfw, "load_sfw_classifier", lambda provider: classifier)

    summary = sfw.classify_images([broken_path, safe_path], verbose=1)

    output = capsys.readouterr().out
    assert f"{broken_path}: error!" in output
    assert "Errors:" not in output
    assert "NSFW: 0/2 (0.0%)" in output
    assert summary.errors == 1
    assert summary.sfw == 1


def test_empty_input_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Avoid model imports and downloads when no images are discovered."""
    monkeypatch.setattr(
        sfw,
        "load_sfw_classifier",
        lambda provider: pytest.fail("model should remain unloaded"),
    )

    summary = sfw.classify_images([], verbose=3)

    assert summary.total == 0
    assert capsys.readouterr().out == "NSFW: 0/0 (0.0%)\n"


def test_model_load_failure_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report every queued image as failed when the optional runtime cannot load."""
    message = "install them with `uv sync`."

    def fail_loading(provider: sfw.SfwProvider) -> sfw.SfwClassifier:
        raise RuntimeError(message)

    monkeypatch.setattr(sfw, "load_sfw_classifier", fail_loading)

    with pytest.raises(RuntimeError, match="uv sync"):
        sfw.classify_images([Path("one.jpg"), Path("two.jpg")], verbose=1)

    output = capsys.readouterr().out
    assert "loading " not in output
    assert "NSFW: 0/2 (0.0%)" in output
    assert "Errors:" not in output


def test_sfw_parser_matches_tag_file_selection_arguments() -> None:
    """Expose directory, extension, and shared verbosity options."""
    args = cli.build_parser().parse_args(
        ["sfw", "uploads", "--extensions", "jpg,png", "-vv"]
    )

    assert args.directory == Path("uploads")
    assert args.extensions == [".jpg", ".png"]
    assert args.verbose == 1
    assert args.verbose_delta == 2
    assert args.nsfw_model == ["falconsai"]
    assert args.threshold == 0.5
    assert args.vote_threshold == 1
    assert args.grid_depth is None
    assert args.func is cli.sfw


def test_sfw_parser_accepts_comma_delimited_models() -> None:
    """Accept each supported classifier in one ordered list."""
    args = cli.build_parser().parse_args(
        ["sfw", "--nsfw-model", "falconsai,marqo,adamcodd"]
    )

    assert args.nsfw_model == ["falconsai", "marqo", "adamcodd"]


def test_sfw_cli_passes_resolved_verbosity_and_discovered_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass final verbosity explicitly through discovery and classification."""
    stackmap_path = tmp_path / "config.yaml"
    stackmap_path.write_text(f"default: {tmp_path}\nart: {tmp_path / 'art'}\n")
    image_paths = [tmp_path / "one.jpg"]
    calls: list[
        tuple[list[Path], list[sfw.SfwProvider], float, int, int, int]
    ] = []

    def fake_classify_images(
        paths: list[Path],
        *,
        providers: list[sfw.SfwProvider],
        threshold: float,
        vote_threshold: int,
        grid_depth: int,
        verbose: int,
    ) -> sfw.SfwSummary:
        calls.append(
            (paths, providers, threshold, vote_threshold, grid_depth, verbose)
        )
        return sfw.SfwSummary(total=len(paths), sfw=len(paths), nsfw=0, errors=0)

    monkeypatch.setattr(cli, "find_images", lambda *args, **kwargs: image_paths)
    monkeypatch.setattr(sfw, "classify_images", fake_classify_images)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image-tagger",
            "sfw",
            str(tmp_path),
            "--stackmap",
            str(stackmap_path),
            "--nsfw-model",
            "falconsai,adamcodd",
            "--threshold",
            "0.7",
            "--vote-threshold",
            "2",
            "--grid-depth",
            "3",
            "-vv",
        ],
    )

    cli.main()

    assert calls == [
        (image_paths, ["falconsai", "adamcodd"], 0.7, 2, 3, 3)
    ]


class FakeTensor:
    """Provide the tensor methods used by the classifier."""

    shape = (1, 2)

    def unsqueeze(self, dim: int) -> FakeTensor:
        """Return this fake tensor with a batch dimension."""
        return self

    def to(self, device: str) -> FakeTensor:
        """Return this device-independent fake tensor."""
        return self

    def detach(self) -> FakeTensor:
        """Return this fake tensor."""
        return self

    def cpu(self) -> FakeTensor:
        """Return this fake tensor."""
        return self

    def tolist(self) -> list[float]:
        """Return deterministic values."""
        return [0.25, 0.75]


class FakeInputs(dict[str, FakeTensor]):
    """Provide device transfer for processor output."""

    def to(self, device: str) -> FakeInputs:
        """Return this device-independent fake batch."""
        return self


class FakeInferenceMode:
    """Provide an inference context manager."""

    def __enter__(self) -> None:
        """Enter inference mode."""

    def __exit__(self, *args: object) -> None:
        """Exit inference mode."""


class FakeTorch:
    """Provide the small torch surface used by the classifier."""

    @staticmethod
    def inference_mode() -> FakeInferenceMode:
        """Return an inference context manager."""
        return FakeInferenceMode()

    @staticmethod
    def softmax(logits: FakeTensor, dim: int) -> FakeTensor:
        """Return predetermined probabilities."""
        return logits


class FakeProcessor:
    """Build fake model inputs."""

    def __call__(self, **kwargs: Any) -> FakeInputs:
        """Return a fake pixel tensor batch."""
        return FakeInputs(pixel_values=FakeTensor())


class FakeModel:
    """Return fake ViT logits and labels."""

    config = SimpleNamespace(id2label={0: "sfw", 1: "nsfw"})

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        """Return fake logits."""
        return SimpleNamespace(logits=[FakeTensor()])


def test_hugging_face_classifier_maps_model_labels(
    tmp_path: Path,
) -> None:
    """Map the model's normal and nsfw labels to public safety labels."""
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 6)).save(image_path)
    classifier = sfw.HuggingFaceSfwClassifier(
        FakeProcessor(),
        FakeModel(),
        FakeTorch(),
        "cpu",
    )

    with Image.open(image_path) as image:
        result = classifier.classify(
            image,
            input_details={"filepath": str(image_path)},
        )
        stricter_result = classifier.classify(
            image,
            input_details={"filepath": str(image_path)},
            threshold=0.8,
        )

    assert result.label == "NSFW"
    assert result.confidence == 0.75
    assert result.scores == {"SFW": 0.25, "NSFW": 0.75}
    assert result.input_details == {
        "filepath": str(image_path),
        "mode": "RGB",
        "size": [8, 6],
        "tensors": {"pixel_values": [1, 2]},
    }

    assert stricter_result.label == "SFW"
    assert stricter_result.output_details["nsfw_threshold"] == 0.8


class FakeMarqoModel:
    """Return fake Marqo logits and labels."""

    pretrained_cfg = {"label_names": ["NSFW", "SFW"]}

    def __call__(self, inputs: FakeTensor) -> list[FakeTensor]:
        """Return fake logits."""
        return [FakeTensor()]


def test_marqo_classifier_maps_model_labels(tmp_path: Path) -> None:
    """Map Marqo label order to public safety labels."""
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 6)).save(image_path)
    classifier = sfw.MarqoSfwClassifier(
        FakeMarqoModel(),
        lambda image: FakeTensor(),
        FakeTorch(),
        "cpu",
    )

    with Image.open(image_path) as image:
        result = classifier.classify(
            image,
            input_details={"filepath": str(image_path)},
        )
        lenient_result = classifier.classify(
            image,
            input_details={"filepath": str(image_path)},
            threshold=0.2,
        )

    assert result.label == "SFW"
    assert result.confidence == 0.75
    assert result.scores == {"SFW": 0.75, "NSFW": 0.25}
    assert result.input_details == {
        "filepath": str(image_path),
        "mode": "RGB",
        "size": [8, 6],
        "tensor_shape": [1, 2],
    }

    assert lenient_result.label == "NSFW"
    assert lenient_result.output_details["nsfw_threshold"] == 0.2
