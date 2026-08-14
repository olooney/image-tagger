import base64
import csv
import hashlib
import json
import os
import random
import re
import string
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from importlib import resources
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, cast
from urllib.parse import urlsplit

import jinja2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from pydantic import BaseModel, create_model
from send2trash import send2trash
from send2trash.exceptions import TrashPermissionError

from constants import WELCOME_EXTENSIONS
from stackmap import StackMap
from util import (
    Pathish,
    TemporarySeed,
    connect_to_openai,
    display_file_operation,
    make_unique,
    quote_display_path,
)


class ImageTagData(BaseModel):
    """Structured metadata returned by vision models."""

    description: str
    category: str
    genre: str
    tags: list[str]
    filename_already_makes_sense: bool
    filename: str


def image_tag_response_model(categories: list[str]) -> type[BaseModel]:
    """Build the tagging schema for the configured shelf aliases."""
    if not categories:
        raise ValueError("stack map must define at least one non-default shelf for tagging.")
    category_type = cast(Any, Literal)[tuple(categories)]
    return create_model(
        "ConfiguredImageTagData",
        __base__=ImageTagData,
        category=(category_type, ...),
    )


def _format_categories(
    categories: list[str],
    descriptions: Mapping[str, str],
) -> str:
    """Format configured category identifiers and optional guidance."""
    return ", ".join(
        f'"{category}": {descriptions[category]}'
        if category in descriptions
        else f'"{category}"'
        for category in categories
    )


class SameImageJudgement(BaseModel):
    """Structured duplicate survivor judgement returned by vision models."""

    thinking: str
    keep: Literal["left", "right", "both"]


@dataclass(frozen=True)
class ImageSimilarity:
    """A scored pair of potentially duplicate images."""

    score: float
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class ImageDuplicateMatch:
    """Accepted duplicate image match."""

    score: float
    left_path: Path
    right_path: Path
    decision_source: str
    judgement_text: str | None = None
    presented_left_path: Path | None = None
    presented_right_path: Path | None = None


@dataclass(frozen=True)
class DedupeReviewEntry:
    """Describe one duplicate decision in the static review report."""

    left_path: Path
    right_path: Path
    left_image_src: str | None
    right_image_src: str | None
    score: float
    decision_source: str
    judgement_text: str | None
    action: str
    duplicate_side: Literal["left", "right"]


class ImageComparisonMethod(Protocol):
    """Score image pairs; higher scores mean more likely duplicates."""

    def compare(
        self,
        left_images: list[Path],
        right_images: list[Path] | None = None,
        *,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> list[ImageSimilarity]:
        """Return scored image pairs."""
        ...


@contextmanager
def suppress_transformers_progress() -> Iterator[None]:
    """Suppress Transformers progress bars during model loading."""
    from transformers.utils import logging as transformers_logging

    was_enabled = transformers_logging.is_progress_bar_enabled()
    transformers_logging.disable_progress_bar()
    try:
        with redirect_stderr(StringIO()):
            yield
    finally:
        if was_enabled:
            transformers_logging.enable_progress_bar()


IMAGE_PROMPT_TEMPLATE: str = (
    resources.files("image_tagger_data").joinpath("image_prompt.md").read_text()
)
SAME_IMAGE_PROMPT_TEMPLATE: str = (
    resources.files("image_tagger_data")
    .joinpath("same_image_prompt.md")
    .read_text()
)
CLIP_MODEL: str = "openai/clip-vit-base-patch32"

csv_columns: list[str] = [
    "timestamp",
    "status",
    "total_tokens",
    "provider_name",
    "model",
    "original_filepath",
    "original_filename",
    "width",
    "height",
    "category",
    "genre",
    "filename",
    "clean_filename",
    "filename_already_makes_sense",
    "tags",
    "description",
]
DEDUPE_REVIEW_FILENAME: Path = Path("dedupe_review.html")


class VisionModelProvider(Enum):
    """Supported vision model providers."""

    OPENAI = "openai"
    GEMMA = "gemma"
    QWEN = "qwen"


GEMMA_MODEL: str = "gemma4:12b"
OPENAI_MODEL: str = "gpt-5.5"
QWEN_MODEL: str = "qwen3.5:4b"


@dataclass(frozen=True)
class VisionTaskResult:
    """Validated structured vision model response data."""

    data: BaseModel
    model: str
    total_tokens: int

class VisionModelClientAdapter(ABC):
    """Common interface for vision providers."""

    provider_name: str
    model: str

    def __str__(self) -> str:
        """Format the provider for console output."""
        return f"{self.provider_name} ({self.model})"

    def __repr__(self) -> str:
        """Format the provider for debugging."""
        return f"{self.__class__.__name__}(model={self.model!r})"

    @abstractmethod
    def vision_task(
        self,
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run a vision task."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release provider resources."""
        pass

class OpenAIVisionModelClientAdapter(VisionModelClientAdapter):
    """OpenAI vision provider adapter."""

    provider_name = "OpenAI"

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        """Create an OpenAI adapter."""
        self.model = model
        self.client = connect_to_openai()

    def vision_task(
        self,
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run an OpenAI vision request."""
        image_base64_values = (
            [image_base64] if isinstance(image_base64, str) else image_base64
        )
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_value}"},
            }
            for image_value in image_base64_values
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            response_format=response_format,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_content,
                    ],
                },
            ],
        )
        data = response.choices[0].message.parsed
        if data is None:
            raise ValueError("OpenAI response did not match the requested response model.")
        total_tokens = response.usage.total_tokens if response.usage is not None else 0
        return VisionTaskResult(
            data=data,
            model=response.model,
            total_tokens=total_tokens,
        )

    def cleanup(self) -> None:
        """OpenAI cleanup hook."""
        pass


class OllamaVisionModelClientAdapter(VisionModelClientAdapter):
    """Ollama vision provider adapter."""

    provider_name = "Ollama"

    def __init__(self, model: str) -> None:
        """Create an Ollama adapter."""
        import ollama

        self.model = model
        self.client = ollama.Client()

    def vision_task(
        self,
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run an Ollama vision request."""
        image_base64_values = (
            [image_base64] if isinstance(image_base64, str) else image_base64
        )
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_base64_values,
                },
            ],
            format=response_format.model_json_schema(),
            options={
                "temperature": 0,
                "image_min_tokens": 1120,
                "image_max_tokens": 1120,
            },
        )
        message = response.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response did not include JSON content.")
        return VisionTaskResult(
            data=response_format.model_validate_json(content),
            model=response.get("model", self.model),
            total_tokens=response.get("prompt_eval_count", 0)
            + response.get("eval_count", 0),
        )

    def cleanup(self) -> None:
        """Unload the Ollama model."""
        self.client.generate(model=self.model, prompt="", keep_alive=0)


_vision_model_client_adapters: dict[VisionModelProvider, VisionModelClientAdapter] = {}


def get_vision_model_client_adapter(
    provider: VisionModelProvider,
) -> VisionModelClientAdapter:
    """Return a cached adapter for a provider."""
    provider = VisionModelProvider(provider)
    if provider not in _vision_model_client_adapters:
        if provider == VisionModelProvider.OPENAI:
            _vision_model_client_adapters[provider] = OpenAIVisionModelClientAdapter()
        elif provider == VisionModelProvider.GEMMA:
            _vision_model_client_adapters[provider] = OllamaVisionModelClientAdapter(
                GEMMA_MODEL
            )
        elif provider == VisionModelProvider.QWEN:
            _vision_model_client_adapters[provider] = OllamaVisionModelClientAdapter(
                QWEN_MODEL
            )
        else:
            raise ValueError(f"Unsupported vision model provider: {provider.value}")
    return _vision_model_client_adapters[provider]


def clean_filename(filename: str) -> str:
    """Clean up a suggested filename."""
    filename = filename.lower()
    filename = re.sub(r"^[^a-zA-Z_]+", "", filename)  # strip leading whitespace
    filename = re.sub(r"[\s_-]+", "_", filename)  # whitespace to underscore
    filename = re.sub(r"[^a-zA-Z0-9_.]", "", filename)  # strip special characters
    filename = re.sub(r"[\s_-]*\.+", ".", filename)  # whitespace before dot

    return filename


def fix_extension(current_filename: str, suggested_filename: str) -> str:
    """Force a suggested filename to keep its original extension."""
    current_path = Path(current_filename)
    suggested_path = Path(suggested_filename)
    if current_path.suffix.lower() != suggested_path.suffix.lower():
        suggested_path = suggested_path.with_suffix(current_path.suffix)
    return suggested_path.name


def path_name_ext(path: Pathish) -> tuple[str, str, str]:
    """Split a path into directory, stem, and extension."""
    image_path = Path(path)
    return (os.fspath(image_path.parent) + os.sep, image_path.stem, image_path.suffix)


def scramble(filename: str) -> str:
    """Hash a filename to obscure it for testing."""
    with TemporarySeed(seed=hash(filename)):
        return "".join(random.sample(string.ascii_letters, k=8))


def resize_image_to_fit(
    image: Image.Image | Pathish,
    max_dimension: int = 512,
) -> Image.Image:
    """Resize an image to fit inside a square."""
    # read from disk if given a filename
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    original_width, original_height = image.size

    # Determine which dimension is larger and calculate scaling factor
    if max(original_width, original_height) > max_dimension:
        if original_width > original_height:
            scaling_factor = max_dimension / original_width
        else:
            scaling_factor = max_dimension / original_height

        # calculate new dimensions from the chosen scale
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # resize with high-quality downsampling
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


def base64_encode_image(image: Image.Image | Pathish) -> str:
    """Encode an image as base64 PNG data."""

    if not isinstance(image, Image.Image):
        image = Image.open(os.fspath(image))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_data = buffer.getvalue()
    base64_encoded_bytes = base64.b64encode(byte_data)

    return base64_encoded_bytes.decode("utf-8")


class ClipImageComparisonMethod:
    """Compare images with normalized CLIP image embeddings."""

    def __init__(self, model: str = CLIP_MODEL) -> None:
        """Create a CLIP comparison method."""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.model = model
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with suppress_transformers_progress():
            self.processor: Any = CLIPProcessor.from_pretrained(self.model)
            clip_client: Any = CLIPModel.from_pretrained(self.model)
        self.client = clip_client.to(self.device)
        self.client.eval()

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed PIL images as normalized CLIP vectors."""
        inputs = self.processor(
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with self.torch.no_grad():
            output = self.client.get_image_features(**inputs)
            features = output.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.detach().cpu().to(self.torch.float64).numpy()

    def compare(
        self,
        left_images: list[Path],
        right_images: list[Path] | None = None,
        *,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> list[ImageSimilarity]:
        """Return CLIP similarity scores for image pairs."""
        left_vectors = embed_image_paths(
            left_images,
            clip_adapter=self,
            batch_size=batch_size,
            verbose=verbose,
        )
        if right_images is None:
            similarity_matrix = left_vectors @ left_vectors.T
            upper_i, upper_j = np.triu_indices_from(similarity_matrix, k=1)
            return [
                ImageSimilarity(
                    score=float(similarity_matrix[i, j]),
                    left_path=left_images[int(i)],
                    right_path=left_images[int(j)],
                )
                for i, j in zip(upper_i, upper_j, strict=True)
            ]

        right_vectors = embed_image_paths(
            right_images,
            clip_adapter=self,
            batch_size=batch_size,
            verbose=verbose,
        )
        similarity_matrix = left_vectors @ right_vectors.T
        similarities: list[ImageSimilarity] = []
        for left_index, left_path in enumerate(left_images):
            for right_index, right_path in enumerate(right_images):
                if left_path == right_path:
                    continue
                similarities.append(
                    ImageSimilarity(
                        score=float(similarity_matrix[left_index, right_index]),
                        left_path=left_path,
                        right_path=right_path,
                    )
                )
        return similarities


def embed_image_paths(
    image_paths: list[Path],
    clip_adapter: ClipImageComparisonMethod | None = None,
    batch_size: int = 32,
    verbose: int = 1,
) -> np.ndarray:
    """Embed image paths as normalized CLIP vectors."""
    clip = clip_adapter or ClipImageComparisonMethod()
    vectors: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images: list[Image.Image] = []

        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB").copy())

        vectors.append(clip.embed_images(batch_images))
        if verbose >= 2:
            complete_count = min(start + batch_size, len(image_paths))
            print(f"embedded {complete_count}/{len(image_paths)}")

    if not vectors:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(vectors)


def image_dimensions(image_path: Pathish) -> tuple[int, int]:
    """Return image width and height."""
    with Image.open(image_path) as image:
        return image.size


def automatic_duplicate_survivor(left_path: Path, right_path: Path) -> tuple[Path, Path]:
    """Choose the kept and duplicate paths for automatic matches."""
    left_width, left_height = image_dimensions(left_path)
    right_width, right_height = image_dimensions(right_path)
    left_pixels = left_width * left_height
    right_pixels = right_width * right_height
    if left_pixels > right_pixels:
        return left_path, right_path
    if right_pixels > left_pixels:
        return right_path, left_path
    if left_path.name <= right_path.name:
        return left_path, right_path
    return right_path, left_path


def format_image_detail(image_path: Pathish) -> str:
    """Format an image path with dimensions for verbose output."""
    width, height = image_dimensions(image_path)
    return f"{quote_display_path(image_path)} ({width}x{height})"


def judge_same_image_match(
    left_path: Pathish,
    right_path: Pathish,
    *,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    client_adapter: VisionModelClientAdapter | None = None,
    max_dimension: int = 768,
) -> SameImageJudgement:
    """Ask a vision model whether two images are the same source image."""
    owned_adapter = client_adapter is None
    if client_adapter is None:
        client_adapter = get_vision_model_client_adapter(provider)

    try:
        left_width, left_height = image_dimensions(left_path)
        right_width, right_height = image_dimensions(right_path)
        prompt = SAME_IMAGE_PROMPT_TEMPLATE.format(
            left_filename=Path(left_path).name,
            left_width=left_width,
            left_height=left_height,
            right_filename=Path(right_path).name,
            right_width=right_width,
            right_height=right_height,
        )
        left_image = resize_image_to_fit(left_path, max_dimension=max_dimension)
        right_image = resize_image_to_fit(right_path, max_dimension=max_dimension)
        image_base64 = [base64_encode_image(left_image), base64_encode_image(right_image)]
        response = client_adapter.vision_task(
            image_base64,
            prompt,
            SameImageJudgement,
        )
        return SameImageJudgement.model_validate(response.data.model_dump())
    finally:
        if owned_adapter:
            client_adapter.cleanup()


def dedupe_image_matches(
    left_images: Iterable[Pathish],
    right_images: Iterable[Pathish] | None = None,
    *,
    automatic_threshold: float = 0.99,
    llm_threshold: float = 0.9,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    batch_size: int = 32,
    comparison_method: ImageComparisonMethod | None = None,
    client_adapter: VisionModelClientAdapter | None = None,
    rejected_llm_matches: list[ImageDuplicateMatch] | None = None,
    verbose: int = 1,
) -> list[ImageDuplicateMatch]:
    """Return accepted duplicate matches from image similarity scores."""
    left_paths = sorted(Path(path) for path in left_images)
    right_paths = None if right_images is None else sorted(Path(path) for path in right_images)
    if right_paths is None and len(left_paths) < 2:
        return []
    if right_paths is not None and (not left_paths or not right_paths):
        return []
    method = comparison_method or ClipImageComparisonMethod()
    similarities = method.compare(
        left_paths,
        right_paths,
        batch_size=batch_size,
        verbose=verbose,
    )

    matches: list[ImageDuplicateMatch] = []
    planned_removals: set[Path] = set()
    borderline_similarities: list[ImageSimilarity] = []
    for similarity in similarities:
        if similarity.score >= automatic_threshold:
            kept_path, duplicate_path = automatic_duplicate_survivor(
                similarity.left_path,
                similarity.right_path,
            )
            if {kept_path, duplicate_path} & planned_removals:
                continue
            matches.append(
                ImageDuplicateMatch(
                    score=similarity.score,
                    left_path=kept_path,
                    right_path=duplicate_path,
                    decision_source="clip",
                )
            )
            planned_removals.add(duplicate_path)
        elif similarity.score >= llm_threshold:
            borderline_similarities.append(similarity)

    for similarity in borderline_similarities:
        if {similarity.left_path, similarity.right_path} & planned_removals:
            continue
        if verbose >= 2:
            print(
                f"LLM duplicate candidate {similarity.score * 100:0.2f}%:\n"
                f"  left: {format_image_detail(similarity.left_path)}\n"
                f"  right: {format_image_detail(similarity.right_path)}"
            )
        judgement = judge_same_image_match(
            similarity.left_path,
            similarity.right_path,
            provider=provider,
            client_adapter=client_adapter,
        )
        if verbose >= 2:
            print(f"  keep: {judgement.keep}")
            if judgement.thinking:
                print(f"  reason: {judgement.thinking}")
        if judgement.keep == "both":
            if rejected_llm_matches is not None:
                rejected_llm_matches.append(
                    ImageDuplicateMatch(
                        score=similarity.score,
                        left_path=similarity.left_path,
                        right_path=similarity.right_path,
                        decision_source="llm",
                        judgement_text=judgement.thinking,
                        presented_left_path=similarity.left_path,
                        presented_right_path=similarity.right_path,
                    )
                )
            continue
        kept_path = similarity.left_path if judgement.keep == "left" else similarity.right_path
        duplicate_path = similarity.right_path if judgement.keep == "left" else similarity.left_path
        if kept_path != duplicate_path:
            matches.append(
                ImageDuplicateMatch(
                    score=similarity.score,
                    left_path=kept_path,
                    right_path=duplicate_path,
                    decision_source="llm",
                    judgement_text=judgement.thinking,
                    presented_left_path=similarity.left_path,
                    presented_right_path=similarity.right_path,
                )
            )
            planned_removals.add(duplicate_path)

    return sorted(matches, key=lambda match: match.score, reverse=True)


def tag_image(
    filepath: Pathish,
    client_adapter: VisionModelClientAdapter,
    prompt_template: str = IMAGE_PROMPT_TEMPLATE,
    categories: list[str] | None = None,
    category_descriptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Tag a single image with a vision model."""
    # handle local or remote images
    filepath_string = os.fspath(filepath)
    if filepath_string.startswith("http"):
        url = filepath_string
        filename = urlsplit(url).path.split("/")[-1]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = resize_image_to_fit(image)
    else:
        image_path = Path(filepath)
        filename = image_path.name
        image = resize_image_to_fit(image_path)

    base64_image_data = base64_encode_image(image)

    # run the tagging vision task and record the time it took
    prompt = prompt_template.format(
        filename=filename,
        categories=_format_categories(categories or [], category_descriptions or {}),
    )
    vision_start_time = time.perf_counter()
    response_format = (
        image_tag_response_model(categories) if categories is not None else ImageTagData
    )
    response = client_adapter.vision_task(base64_image_data, prompt, response_format)
    vision_duration = time.perf_counter() - vision_start_time
    data = response.data.model_dump()

    # clean up the suggested filename and fix the extension if necessary
    suggested_filename_value = data["filename"]
    if not isinstance(suggested_filename_value, str):
        raise ValueError("Vision response filename must be a string.")
    suggested_filename = clean_filename(suggested_filename_value)
    suggested_filename_fixed = fix_extension(filename, suggested_filename)

    # format the results
    data["clean_filename"] = suggested_filename_fixed
    data["original_filepath"] = filepath_string
    data["original_filename"] = filename
    data["total_tokens"] = response.total_tokens
    data["provider_name"] = client_adapter.provider_name
    data["model"] = response.model
    data["width"] = image.size[0]
    data["height"] = image.size[1]
    data["vision_duration"] = vision_duration

    return data


def tag_images(
    filepaths: Iterable[Pathish],
    output_filename: Pathish,
    retry_errors: bool = False,
    verbose: int = 1,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    instructions_filename: Pathish | None = None,
    categories: list[str] | None = None,
    category_descriptions: Mapping[str, str] | None = None,
) -> None:
    """Tag images and write metadata rows."""
    output_path = Path(output_filename)
    client_adapter = get_vision_model_client_adapter(provider)
    if instructions_filename is None:
        prompt_template = IMAGE_PROMPT_TEMPLATE
    else:
        prompt_template = Path(instructions_filename).read_text(encoding="utf-8")
    if verbose >= 1:
        print(f"Using {client_adapter}")
    file_already_exists = output_path.exists()
    mode = "a" if file_already_exists else "w"

    try:
        with output_path.open(mode, newline="", encoding="utf-8") as csv_file:
            columns = csv_columns
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            if not file_already_exists:
                writer.writeheader()

            vision_durations = []

            for index, filepath in enumerate(filepaths):
                row_start_time = time.perf_counter()

                try:
                    # run the model and normalize row fields for CSV output
                    row = tag_image(
                        filepath,
                        client_adapter,
                        prompt_template,
                        categories,
                        category_descriptions,
                    )
                    duration = row.pop("vision_duration")
                    vision_durations.append(duration)
                    row["tags"] = ";".join(tag.lower().strip() for tag in row["tags"])
                    row.update(
                        {"timestamp": datetime.now().isoformat(), "status": "ok"}
                    )
                    writer.writerow(row)
                    csv_file.flush()

                    if verbose == 0:
                        print(".", end=("\n" if (index + 1) % 100 == 0 else ""))
                    elif verbose == 1:
                        average_durations = (
                            vision_durations[1:]
                            if len(vision_durations) > 2
                            else vision_durations
                        )
                        average_duration = sum(average_durations) / len(
                            average_durations
                        )
                        print(
                            f"{row['timestamp']} {row['original_filename']} -> "
                            f"{row['clean_filename']}: {row['category']} {row['genre']} {row['status']} "
                            f"{duration:0.2f}s avg {average_duration:0.2f}s"
                        )
                    elif verbose >= 2:
                        print(repr(row))
                except KeyboardInterrupt:
                    if verbose >= 1:
                        print("\nInterrupted; cleaning up...")
                    raise
                except Exception:
                    error_message = traceback.format_exc()
                    duration = time.perf_counter() - row_start_time

                    if verbose == 1:
                        print("e", end=("\n" if (index + 1) % 100 == 0 else ""))
                    elif verbose == 2:
                        original_filename = Path(filepath).name
                        print(
                            f"{datetime.now().isoformat()} {original_filename} -> "
                            f"<none> error {duration:0.2f}s"
                        )
                    elif verbose >= 3:
                        print(error_message)

                    writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "original_filepath": filepath,
                            "status": "error",
                            "description": error_message,
                        }
                    )
    finally:
        client_adapter.cleanup()


def previously_tagged_filenames(metadata_filename: Pathish) -> set[str]:
    """Return filenames already tagged successfully."""
    metadata_path = Path(metadata_filename)
    if not metadata_path.exists():
        return set()

    tagged_filenames: set[str] = set()
    with metadata_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("status") != "ok":
                continue

            for column in ["original_filename", "clean_filename"]:
                value = row.get(column)
                if not value:
                    continue
                tagged_filenames.add(Path(value).name)

    return tagged_filenames


def find_images(
    dirs: Pathish | Iterable[Pathish],
    max_days_old: float | None = None,
    metadata_filename: Pathish | None = None,
    extension_filter: Iterable[str] | None = WELCOME_EXTENSIONS,
) -> list[Path]:
    """Find untagged image files in directories recursively."""
    if max_days_old is None:
        max_days_old = float("Inf")

    if isinstance(dirs, (str, os.PathLike)):
        directories = [dirs]
    else:
        directories = dirs

    tagged_filenames = (
        previously_tagged_filenames(metadata_filename)
        if metadata_filename is not None
        else set()
    )
    allowed_extensions = (
        {extension.lower() for extension in extension_filter}
        if extension_filter is not None
        else None
    )
    current_time = time.time()

    filepaths: list[Path] = []
    for directory in directories:
        directory_path = Path(cast("Any", directory))
        for filepath in directory_path.rglob("*"):
            if not filepath.is_file():
                continue
            if (current_time - filepath.stat().st_mtime) >= max_days_old * 86400:
                continue
            if allowed_extensions is not None:
                if filepath.suffix.lower() not in allowed_extensions:
                    continue
            if filepath.name in tagged_filenames:
                continue
            filepaths.append(filepath)

    return filepaths


def dedupe_images(
    directory: Pathish,
    *,
    automatic_threshold: float = 0.99,
    llm_threshold: float = 0.9,
    verbose: int = 1,
    dry_run: bool = False,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    batch_size: int = 32,
) -> list[ImageDuplicateMatch]:
    """Send duplicate images from a directory to the recycle bin."""
    directory_path = Path(directory)
    image_paths = sorted(find_images(directory_path))
    if verbose == 1:
        print(f"working in {quote_display_path(directory_path)}")
    rejected_llm_matches: list[ImageDuplicateMatch] = []
    matches = dedupe_image_matches(
        image_paths,
        automatic_threshold=automatic_threshold,
        llm_threshold=llm_threshold,
        provider=provider,
        batch_size=batch_size,
        verbose=verbose,
        rejected_llm_matches=rejected_llm_matches,
    )

    removed_paths: set[Path] = set()
    review_entries = [
        _dedupe_review_entry(
            match,
            action="Both images were kept after LLM adjudication.",
        )
        for match in rejected_llm_matches
    ]
    for match in matches:
        duplicate = match.right_path
        kept = match.left_path
        if duplicate in removed_paths or kept in removed_paths:
            continue
        review_entry = _dedupe_review_entry(match, action="")
        action = _duplicate_removal_action(
            review_entry.duplicate_side,
            dry_run=dry_run,
        )
        if verbose >= 1:
            print(
                display_file_operation(
                    "removing duplicate",
                    duplicate,
                    kept,
                    verbose=verbose,
                    relative_to=directory_path,
                ),
                end="",
            )
        try:
            if not dry_run:
                try:
                    send2trash(duplicate)
                except TrashPermissionError:
                    duplicate.unlink()
            removed_paths.add(duplicate)
            if verbose >= 1:
                print("success!")
        except Exception:
            action = _duplicate_removal_action(
                review_entry.duplicate_side,
                dry_run=False,
                removed=False,
            )
            if verbose >= 1:
                print("error!")
            else:
                print(f"error removing {os.fspath(duplicate)!r}!")
            traceback.print_exc()
        review_entries.append(replace(review_entry, action=action))

    review_filename = directory_path / DEDUPE_REVIEW_FILENAME
    generate_dedupe_review(review_entries, review_filename)
    if verbose >= 1:
        print(f"wrote {quote_display_path(review_filename)}")

    return matches


def _dedupe_review_entry(
    match: ImageDuplicateMatch,
    *,
    action: str,
) -> DedupeReviewEntry:
    """Capture report thumbnails before a duplicate can be removed."""
    left_path = match.presented_left_path or match.left_path
    right_path = match.presented_right_path or match.right_path
    duplicate_side: Literal["left", "right"] = (
        "left" if match.right_path == left_path else "right"
    )
    return DedupeReviewEntry(
        left_path=left_path,
        right_path=right_path,
        left_image_src=_thumbnail_data_url(left_path),
        right_image_src=_thumbnail_data_url(right_path),
        score=match.score,
        decision_source=match.decision_source,
        judgement_text=match.judgement_text,
        action=action,
        duplicate_side=duplicate_side,
    )


def _duplicate_removal_action(
    duplicate_side: Literal["left", "right"],
    *,
    dry_run: bool,
    removed: bool = True,
) -> str:
    """Describe the review side selected for duplicate removal."""
    side = duplicate_side.capitalize()
    if not removed:
        return f"{side} image could not be removed as a duplicate."
    if dry_run:
        return f"{side} image would be removed as a duplicate."
    return f"{side} image was removed as a duplicate."


def _thumbnail_data_url(image_path: Pathish) -> str | None:
    """Encode an image as a PNG data URL within a 500-by-750-pixel bound."""
    try:
        with Image.open(image_path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((500, 750), Image.Resampling.LANCZOS)
            return f"data:image/png;base64,{base64_encode_image(thumbnail)}"
    except (OSError, ValueError):
        return None


def generate_dedupe_review(
    entries: Sequence[DedupeReviewEntry],
    output_filename: Pathish,
) -> Path:
    """Render a static HTML report for duplicate-removal decisions."""
    template_text = (
        resources.files("image_tagger_data")
        .joinpath("dedupe_review.html")
        .read_text(encoding="utf-8")
    )
    template = jinja2.Environment(autoescape=True).from_string(template_text)
    output_path = Path(output_filename)
    output_path.write_text(template.render(entries=entries), encoding="utf-8")
    return output_path


def scramble_image_directory(
    input_dir: Pathish,
    output_dir: Pathish,
    max_dimension: int = 512,
) -> None:
    """Copy resized images with scrambled stems."""
    output_path = Path(output_dir)
    for filepath in find_images(input_dir):
        scrambled_name = scramble(filepath.stem)
        new_filepath = output_path / f"{scrambled_name}{filepath.suffix}"
        thumbnail = resize_image_to_fit(filepath, max_dimension)
        thumbnail.save(new_filepath)


def median_image_aspect_ratio(filepaths: Iterable[Pathish]) -> float:
    """Return the median width-to-height ratio for image files."""
    aspect_ratios = image_aspect_ratios(filepaths)
    return median_aspect_ratio(aspect_ratios.values())


def median_aspect_ratio(aspect_ratios: Iterable[float]) -> float:
    """Return the median aspect ratio from ratio values."""
    sorted_aspect_ratios = sorted(aspect_ratios)

    if not sorted_aspect_ratios:
        return 1.0

    midpoint = len(sorted_aspect_ratios) // 2
    if len(sorted_aspect_ratios) % 2:
        return sorted_aspect_ratios[midpoint]
    return (sorted_aspect_ratios[midpoint - 1] + sorted_aspect_ratios[midpoint]) / 2


def image_aspect_ratios(filepaths: Iterable[Pathish]) -> dict[Path, float]:
    """Return width-to-height ratios keyed by image path."""
    aspect_ratios: dict[Path, float] = {}
    for filepath in filepaths:
        image_path = Path(filepath)
        with Image.open(image_path) as image:
            width, height = image.size
        if height > 0:
            aspect_ratios[image_path] = width / height

    return aspect_ratios


def paths_with_mtime(filepaths: Iterable[Pathish]) -> list[tuple[float, Path]]:
    """Return file paths paired with their modification times."""
    return [(Path(filepath).stat().st_mtime, Path(filepath)) for filepath in filepaths]


DEFAULT_WALL_RANDOM_SEED: int = 37


def seeded_wall_sort_key(directory: Path, filepath: Path, seed: int) -> tuple[bytes, str]:
    """Return a stable pseudo-random wall sort key for a file path."""
    relative_filepath = Path(os.path.relpath(filepath, directory)).as_posix()
    key_text = f"{seed}\0{relative_filepath}".encode("utf-8", errors="surrogateescape")
    return hashlib.blake2b(key_text, digest_size=16).digest(), relative_filepath


def singular_wall_title_word(word: str) -> str:
    """Return a simple singular display form for a wall title word."""
    if len(word) > 3 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 1 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def wall_title_from_directory(directory: Pathish) -> str:
    """Return a display title inferred from a wall directory name."""
    directory_name = Path(directory).name.replace("_", " ").strip()
    title_words = [
        singular_wall_title_word(word)
        for word in re.split(r"\s+", directory_name.casefold())
        if word
    ]
    if not title_words:
        return "Image Wall"
    return f"{' '.join(title_words).title()} Wall"

WALL_TITLE_TEMPLATE = """{clean_filename} ({width}x{height})
Category: {category}
Genre: {genre}
Tags: {tags}

{description}
"""


def wall_metadata_titles(metadata_filename: Pathish | None) -> dict[str, str]:
    """Return image metadata title text keyed by possible filenames."""
    if metadata_filename is None:
        return {}

    metadata_path = Path(metadata_filename)
    if not metadata_path.is_file():
        return {}

    metadata_df = pd.read_csv(metadata_path)
    titles: dict[str, str] = {}
    for raw_item in metadata_df.to_dict("records"):
        item = cast("dict[str, Any]", raw_item)
        if item.get("status") != "ok":
            continue

        item['tags'] = item['tags'].replace(";", ", ")
        title = WALL_TITLE_TEMPLATE.format(**item)

        if not title:
            continue

        for column in ["original_filepath", "original_filename", "clean_filename"]:
            value = item.get(column, "")
            if value and isinstance(value, str):
                titles[Path(value).name] = title
                titles[os.fspath(Path(value))] = title

    return titles


def generate_wall(
    directory: Pathish,
    output_filename: Pathish | None = None,
    metadata_filename: Pathish | None = None,
    order: str = "random",
    seed: int | None = None,
    title: str | None = None,
    verbose: int = 1,
) -> Path:
    """Generate a static image wall HTML file."""
    directory_path = Path(directory)
    output_path = (
        Path(output_filename) if output_filename is not None else directory_path / "index.html"
    )
    filepaths = find_images(directory_path)
    if order == "name":
        filepaths.sort(key=lambda filepath: filepath.name.casefold())
    elif order == "date":
        filepaths = [
            filepath
            for _, filepath in sorted(paths_with_mtime(filepaths), reverse=True)
        ]
    elif order == "random":
        if seed is None:
            random.shuffle(filepaths)
        else:
            filepaths.sort(
                key=lambda filepath: seeded_wall_sort_key(directory_path, filepath, seed)
            )
    else:
        raise ValueError(f"Unsupported wall order: {order}")
    aspect_ratios = image_aspect_ratios(filepaths)
    aspect_ratio = median_aspect_ratio(aspect_ratios.values())
    cell_width = 200
    cell_height = round(cell_width / aspect_ratio)
    metadata_titles = wall_metadata_titles(metadata_filename)
    items = [
        {
            "src": Path(os.path.relpath(filepath, output_path.parent)).as_posix(),
            "alt": filepath.name,
            "title": metadata_titles.get(
                os.fspath(filepath),
                metadata_titles.get(filepath.name, filepath.name),
            ),
            "is_double_wide": aspect_ratios.get(filepath, aspect_ratio)
            > 1.8 * aspect_ratio,
        }
        for filepath in filepaths
    ]

    template_text = (
        resources.files("image_tagger_data")
        .joinpath("wall_template.html")
        .read_text(encoding="utf-8")
    )
    template = jinja2.Environment(autoescape=True).from_string(template_text)
    output = template.render(
        items=items,
        cell_width=cell_width,
        cell_height=cell_height,
        wall_title=title or wall_title_from_directory(directory_path),
    )
    output_path.write_text(output, encoding="utf-8")
    if verbose >= 1:
        print(f"wrote {quote_display_path(output_path)}")
    return output_path


def prune_metadata_rows(
    csv_filename: Pathish,
    *,
    verbose: int = 1,
    dry_run: bool = False,
) -> int:
    """Remove metadata rows whose image files are no longer present."""
    csv_path = Path(csv_filename)
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
        item = cast("dict[str, Any]", raw_item)
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

        # new filename
        new_filename = row["clean_filename"]
        target = source.with_name(new_filename)

        # old filename
        if not source.is_file():
            if verbose >= 2:
                print(f"source file {os.fspath(source)!r} is missing!")
            if verbose >= 1 and not target.is_file():
                print(
                    f"both source file {os.fspath(source)!r} and {os.fspath(target)!r} are missing!"
                )
            continue

        # check for no-op
        if target == source:
            if verbose >= 2:
                print(f"no rename necessary for {source!r}")
            continue

        # ensure extension matches
        if source.suffix.lower() != target.suffix.lower():
            if verbose >= 1:
                print(
                    f"Mismatched file extensions between {os.fspath(source)!r} and {os.fspath(target)!r}; skipping rename!"
                )
            continue

        # check for name collisions
        if target.is_file():
            if verbose >= 1:
                print(f"target {os.fspath(target)!r} already exists!")
            target = Path(make_unique(target))
            if verbose >= 1:
                print(f"proceeding with target {os.fspath(target)!r}.")

        # actually perform the file rename
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
            traceback.print_exc()

    if metadata_updated:
        metadata_df.to_csv(csv_path, index=False)


def append_shelved_metadata(
    row: pd.Series[Any],
    target: Path,
    source_metadata_path: Path,
) -> None:
    """Append shelved image metadata to the target directory metadata file."""
    target_metadata_path = target.parent / source_metadata_path.name
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
) -> None:
    """Move images into category folders, while keeping the current shelf scoped."""
    csv_path = Path(csv_filename)
    if not csv_path.is_file():
        return

    metadata_df = pd.read_csv(csv_path)
    display_directory = stackmap.filename.parent
    current_alias = _stackmap_alias_for_directory(stackmap, csv_path.parent)
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for index, row in metadata_df.iterrows():
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
                print(f"category {category!r} is not a configured shelf; skipping {source!r}")
            continue
        if current_alias is not None and category == current_alias:
            if verbose >= 2:
                print(f"already in correct shelf {category!r}; skipping {source!r}")
            continue

        target_directory = stackmap.directory_for(category)
        if target_directory is None:
            if verbose >= 1:
                print(f"category {category!r} is not a configured shelf; skipping {source!r}")
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
            traceback.print_exc()

    if not dry_run:
        prune_metadata_rows(csv_path, verbose=verbose)


def generate_gallery(
    csv_filename: Pathish,
    output_filename: Pathish,
    verbose: int = 1,
) -> None:
    """Generate a static gallery HTML file."""
    # read the metadata and prepare for merge
    csv_path = Path(csv_filename)
    output_path = Path(output_filename)
    metadata_df = pd.read_csv(csv_path)
    metadata_df = metadata_df[metadata_df["status"] == "ok"]
    items: list[dict[str, Any]] = []
    for raw_item in metadata_df.to_dict("records"):
        item = cast("dict[str, Any]", raw_item)
        original_path = Path(item["original_filepath"])
        clean_filename = item.get("clean_filename", "")
        clean_path = (
            original_path.with_name(clean_filename)
            if clean_filename and isinstance(clean_filename, str)
            else None
        )
        if clean_path is not None and clean_path.is_file():
            image_path = clean_path
        elif original_path.is_file():
            image_path = original_path
        else:
            continue

        image_src = os.path.relpath(image_path, output_path.parent)
        item["image_src"] = Path(image_src).as_posix()
        items.append(item)

    first_item = items[0] if items else {}
    provider_name = str(first_item.get("provider_name", "")).strip()
    model = str(first_item.get("model", "")).strip()
    for item in items:
        item["formatted_timestamp"] = datetime.fromisoformat(
            item["timestamp"]
        ).strftime("%m/%d/%y %I:%M %p")
        item["tags"] = [tag.strip() for tag in item["tags"].split(";")]
        notes = item.get("notes", "")
        # filter out the NaNs that pandas uses for missing values.
        item["notes"] = notes if notes and isinstance(notes, str) else ""

    # Render the template with the data
    template_text = (
        resources.files("image_tagger_data").joinpath("template.html").read_text()
    )
    template = jinja2.Template(template_text)
    output = template.render(items=items, provider_name=provider_name, model=model)

    # Save the rendered HTML to a file
    output_path.write_text(output, encoding="utf-8")
