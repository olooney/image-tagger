import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image, ImageOps

from .util import quote_display_path

type SafetyLabel = Literal["SFW", "NSFW"]
type SfwProvider = Literal["falconsai", "marqo", "adamcodd"]

DEFAULT_PROVIDER: SfwProvider = "falconsai"
MODEL_NAMES: dict[SfwProvider, str] = {
    "falconsai": "Falconsai/nsfw_image_detection",
    "marqo": "Marqo/nsfw-image-detection-384",
    "adamcodd": "AdamCodd/vit-base-nsfw-detector",
}


@dataclass(frozen=True)
class SfwClassification:
    """Store one image safety classification."""

    label: SafetyLabel
    confidence: float
    scores: dict[SafetyLabel, float]
    input_details: dict[str, Any]
    output_details: dict[str, Any]


@dataclass(frozen=True)
class SfwSummary:
    """Store aggregate safety classification counts."""

    total: int
    sfw: int
    nsfw: int
    errors: int


class SfwClassifier(Protocol):
    """Classify image safety."""

    def classify(
        self,
        image: Image.Image,
        *,
        input_details: dict[str, Any],
        threshold: float = 0.5,
    ) -> SfwClassification:
        """Classify one prepared image region."""
        ...


class HuggingFaceSfwClassifier:
    """Run a Hugging Face ViT image classifier."""

    def __init__(
        self,
        processor: Any,
        model: Any,
        torch: Any,
        device: str,
    ) -> None:
        """Create a loaded classifier."""
        self.processor = processor
        self.model = model
        self.torch = torch
        self.device = device

    def classify(
        self,
        image: Image.Image,
        *,
        input_details: dict[str, Any],
        threshold: float = 0.5,
    ) -> SfwClassification:
        """Classify one image region and retain JSON-safe diagnostic details."""
        input_details = {
            **input_details,
            "mode": image.mode,
            "size": list(image.size),
        }
        inputs = self.processor(images=image, return_tensors="pt")
        input_details["tensors"] = {
            name: list(value.shape) for name, value in inputs.items()
        }
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        logits = outputs.logits[0]
        probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()

        scores: dict[SafetyLabel, float] = {"SFW": 0.0, "NSFW": 0.0}
        raw_scores: dict[str, float] = {}
        for index, probability in enumerate(probabilities):
            raw_label = str(self.model.config.id2label[index]).lower()
            raw_scores[raw_label] = probability
            if raw_label in {"normal", "sfw"}:
                scores["SFW"] = probability
            elif raw_label == "nsfw":
                scores["NSFW"] = probability

        if "nsfw" not in raw_scores or not raw_scores.keys() & {"normal", "sfw"}:
            raise ValueError(
                "Model labels must include nsfw and either normal or sfw; "
                f"received {sorted(raw_scores)}."
            )
        label: SafetyLabel = "NSFW" if scores["NSFW"] >= threshold else "SFW"
        output_details: dict[str, Any] = {
            "logits": logits.detach().cpu().tolist(),
            "scores": raw_scores,
            "nsfw_threshold": threshold,
            "label": label,
        }
        return SfwClassification(
            label=label,
            confidence=scores[label],
            scores=scores,
            input_details=input_details,
            output_details=output_details,
        )


class MarqoSfwClassifier:
    """Run the Marqo timm image classifier."""

    def __init__(
        self,
        model: Any,
        transform: Any,
        torch: Any,
        device: str,
    ) -> None:
        """Create a loaded classifier."""
        self.model = model
        self.transform = transform
        self.torch = torch
        self.device = device

    def classify(
        self,
        image: Image.Image,
        *,
        input_details: dict[str, Any],
        threshold: float = 0.5,
    ) -> SfwClassification:
        """Classify one image region and retain JSON-safe diagnostic details."""
        input_details = {
            **input_details,
            "mode": image.mode,
            "size": list(image.size),
        }
        inputs = self.transform(image).unsqueeze(0).to(self.device)
        input_details["tensor_shape"] = list(inputs.shape)
        with self.torch.inference_mode():
            logits = self.model(inputs)[0]
        probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()
        raw_labels = self.model.pretrained_cfg["label_names"]
        raw_scores = {
            str(label): probability
            for label, probability in zip(raw_labels, probabilities)
        }
        scores: dict[SafetyLabel, float] = {
            "SFW": raw_scores["SFW"],
            "NSFW": raw_scores["NSFW"],
        }
        label: SafetyLabel = "NSFW" if scores["NSFW"] >= threshold else "SFW"
        return SfwClassification(
            label=label,
            confidence=scores[label],
            scores=scores,
            input_details=input_details,
            output_details={
                "logits": logits.detach().cpu().tolist(),
                "scores": raw_scores,
                "nsfw_threshold": threshold,
                "label": label,
            },
        )


def load_sfw_classifier(
    provider: SfwProvider = DEFAULT_PROVIDER,
) -> SfwClassifier:
    """Load the selected classifier runtime and model on demand."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "SFW classification requires the project's ML dependencies; "
            "install them with `uv sync`."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAMES[provider]
    if provider == "marqo":
        try:
            from timm import create_model
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as error:
            raise RuntimeError(
                "The Marqo SFW provider requires timm; install it with `uv sync`."
            ) from error
        model = create_model(f"hf_hub:{model_name}", pretrained=True).to(device)
        model.eval()
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)
        return MarqoSfwClassifier(model, transform, torch, device)

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as error:
        raise RuntimeError(
            "The AdamCodd SFW provider requires transformers; "
            "install it with `uv sync`."
        ) from error
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    return HuggingFaceSfwClassifier(processor, model, torch, device)


def default_grid_depth(image_size: tuple[int, int]) -> int:
    """Choose an automatic grid depth from the longest image side."""
    max_side = max(image_size)
    if max_side < 500:
        return 0
    if max_side <= 2000:
        return 1
    return 2


def _pad_to_size(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Center an image on a black canvas without resizing it."""
    target_width, target_height = target_size
    horizontal_padding = target_width - image.width
    vertical_padding = target_height - image.height
    if horizontal_padding < 0 or vertical_padding < 0:
        raise ValueError("Padding target must contain the source image.")
    return ImageOps.expand(
        image,
        border=(
            horizontal_padding // 2,
            vertical_padding // 2,
            horizontal_padding - horizontal_padding // 2,
            vertical_padding - vertical_padding // 2,
        ),
        fill="black",
    )


def _grid_positions(length: int, tile_size: int, grid_size: int) -> list[int]:
    """Place grid positions evenly from the first to last valid offset."""
    span = length - tile_size
    return [round(index * span / (grid_size - 1)) for index in range(grid_size)]


def _classify_with_tiles(
    filepath: Path,
    classifier: SfwClassifier,
    *,
    threshold: float,
    grid_depth: int | None,
    scan_results: list[SfwClassification] | None = None,
) -> SfwClassification:
    """Classify a padded gestalt, then progressively finer square grids."""
    with Image.open(filepath) as source_image:
        image = source_image.convert("RGB")
    original_size = image.size
    resolved_grid_depth = (
        default_grid_depth(original_size) if grid_depth is None else grid_depth
    )

    square_size = max(image.size)
    gestalt = _pad_to_size(image, (square_size, square_size))
    try:
        best = classifier.classify(
            gestalt,
            threshold=threshold,
            input_details={
                "filepath": str(filepath),
                "original_size": list(original_size),
                "scan": "gestalt",
                "grid_size": 0,
            },
        )
        if scan_results is not None:
            scan_results.append(best)
    finally:
        gestalt.close()
    if best.label == "NSFW":
        image.close()
        return best

    for depth in range(1, resolved_grid_depth + 1):
        grid_size = depth + 1
        tile_size = (max(original_size) + grid_size - 1) // grid_size
        padded = _pad_to_size(
            image,
            (
                max(image.width, tile_size),
                max(image.height, tile_size),
            ),
        )
        x_positions = _grid_positions(padded.width, tile_size, grid_size)
        y_positions = _grid_positions(padded.height, tile_size, grid_size)
        grid_best = best
        try:
            for row, top in enumerate(y_positions):
                for column, left in enumerate(x_positions):
                    box = (left, top, left + tile_size, top + tile_size)
                    tile = padded.crop(box)
                    try:
                        result = classifier.classify(
                            tile,
                            threshold=threshold,
                            input_details={
                                "filepath": str(filepath),
                                "original_size": list(original_size),
                                "scan": "tile",
                                "grid_size": grid_size,
                                "tile": [row, column],
                                "crop_box": list(box),
                            },
                        )
                        if scan_results is not None:
                            scan_results.append(result)
                    finally:
                        tile.close()
                    if result.scores["NSFW"] > grid_best.scores["NSFW"]:
                        grid_best = result
        finally:
            padded.close()
        best = grid_best
        if best.label == "NSFW":
            break

    image.close()
    return best


def _configure_model_progress(*, enabled: bool) -> None:
    """Configure dependency progress bars for model loading."""
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
    from transformers.utils import logging as transformers_logging

    if enabled:
        enable_progress_bars()
        transformers_logging.enable_progress_bar()
    else:
        disable_progress_bars()
        transformers_logging.disable_progress_bar()


def _format_classification(
    filepath: Path,
    *,
    label: SafetyLabel,
    confidence: float,
    votes: int,
    provider_count: int,
) -> str:
    """Format one aggregate classifier result for the command line."""
    vote_text = f" votes={votes}" if provider_count > 1 else ""
    return (
        f"{quote_display_path(filepath)}: label={label}{vote_text} "
        f"confidence={confidence:.2%}"
    )


def _format_provider_classification(
    provider: SfwProvider,
    classification: SfwClassification,
) -> str:
    """Format one provider result for verbose output."""
    return (
        f"{MODEL_NAMES[provider]}: label={classification.label} "
        f"confidence={classification.confidence:.2%}"
    )


def _format_ratio(label: str, count: int, total: int) -> str:
    """Format a summary count and percentage."""
    percentage = count / total if total else 0.0
    return f"{label}: {count}/{total} ({percentage:.1%})"


def _print_summary(summary: SfwSummary) -> None:
    """Print aggregate NSFW statistics."""
    print(_format_ratio("NSFW", summary.nsfw, summary.total))


def classify_images(
    filepaths: list[Path],
    *,
    providers: list[SfwProvider] | None = None,
    threshold: float = 0.5,
    vote_threshold: int = 1,
    grid_depth: int | None = None,
    verbose: int = 1,
) -> SfwSummary:
    """Classify images and print results according to CLI verbosity."""
    providers = [DEFAULT_PROVIDER] if providers is None else providers
    if not providers:
        raise ValueError("At least one NSFW model is required.")
    if not 1 <= vote_threshold <= len(providers):
        raise ValueError(
            f"Vote threshold must be between 1 and {len(providers)}."
        )
    if not filepaths:
        summary = SfwSummary(total=0, sfw=0, nsfw=0, errors=0)
        _print_summary(summary)
        return summary

    classifiers: dict[SfwProvider, SfwClassifier] = {}
    _configure_model_progress(enabled=verbose >= 2)
    try:
        for provider in providers:
            model_name = MODEL_NAMES[provider]
            if verbose >= 2:
                print(f"loading {model_name} ...", end="", flush=True)
            classifiers[provider] = load_sfw_classifier(provider)
            if verbose >= 2:
                print("success!")
    except Exception:
        if verbose >= 2:
            print("error!")
        summary = SfwSummary(
            total=len(filepaths),
            sfw=0,
            nsfw=0,
            errors=len(filepaths),
        )
        _print_summary(summary)
        raise

    sfw_count = 0
    nsfw_count = 0
    error_count = 0
    for filepath in filepaths:
        try:
            results: dict[SfwProvider, SfwClassification] = {}
            detailed_results: dict[SfwProvider, list[SfwClassification]] = {}
            for provider in providers:
                provider_details: list[SfwClassification] = []
                results[provider] = _classify_with_tiles(
                    filepath,
                    classifiers[provider],
                    threshold=threshold,
                    grid_depth=grid_depth,
                    scan_results=provider_details if verbose >= 3 else None,
                )
                detailed_results[provider] = provider_details
            votes = sum(
                result.label == "NSFW"
                for result in results.values()
            )
            label: SafetyLabel = "NSFW" if votes >= vote_threshold else "SFW"
            if label == "NSFW":
                nsfw_count += 1
                confidence = min(
                    result.confidence
                    for result in results.values()
                    if result.label == "NSFW"
                )
            else:
                sfw_count += 1
                confidence = min(result.scores["SFW"] for result in results.values())

            if verbose >= 2 or (verbose == 1 and label == "NSFW"):
                print(
                    _format_classification(
                        filepath,
                        label=label,
                        confidence=confidence,
                        votes=votes,
                        provider_count=len(providers),
                    )
                )
            if verbose >= 3:
                for provider, provider_results in detailed_results.items():
                    for result in provider_results:
                        print(_format_provider_classification(provider, result))
                        print(
                            f"{MODEL_NAMES[provider]} input:",
                            json.dumps(result.input_details, sort_keys=True),
                        )
                        print(
                            f"{MODEL_NAMES[provider]} output:",
                            json.dumps(result.output_details, sort_keys=True),
                        )
        except KeyboardInterrupt:
            raise
        except Exception:
            error_count += 1
            if verbose >= 1:
                print(f"{quote_display_path(filepath)}: error!")
            if verbose >= 3:
                traceback.print_exc()

    summary = SfwSummary(
        total=len(filepaths),
        sfw=sfw_count,
        nsfw=nsfw_count,
        errors=error_count,
    )
    _print_summary(summary)
    return summary
