import base64
import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from importlib import resources
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel

from .constants import (
    CLIP_MODEL,
    DEFAULT_AUTOMATIC_THRESHOLD,
    DEFAULT_LLM_THRESHOLD,
    EMBEDDING_CACHE_VERSION,
)
from .util import Pathish, quote_display_path
from .vision import (
    VisionModelClientAdapter,
    VisionModelProvider,
    get_vision_model_client_adapter,
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


@dataclass
class EmbeddingCache:
    """Persisted embeddings and image paths confirmed by dedupe."""

    entries: dict[str, tuple[int, int, np.ndarray]]
    reviewed_thresholds: dict[str, tuple[float, float]]
    needs_upgrade: bool = False


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


class ImageEmbedder(Protocol):
    """Embed PIL image batches as normalized vectors."""

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return one vector for each input image."""
        ...


@contextmanager
def suppress_transformers_progress() -> Generator[None]:
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


SAME_IMAGE_PROMPT_TEMPLATE: str = (
    resources.files("image_tagger.data").joinpath("same_image_prompt.md").read_text()
)


def resize_image_to_fit(
    image: Image.Image | Pathish,
    max_dimension: int = 512,
) -> Image.Image:
    """Resize an image to fit inside a square."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    original_width, original_height = image.size

    if max(original_width, original_height) > max_dimension:
        if original_width > original_height:
            scaling_factor = max_dimension / original_width
        else:
            scaling_factor = max_dimension / original_height

        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
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

    def __init__(
        self,
        model: str = CLIP_MODEL,
        cache_filename: Pathish | None = None,
    ) -> None:
        """Create a CLIP comparison method."""
        self.model = model
        self.cache_filename = cache_filename
        self.torch: Any | None = None
        self.processor: Any | None = None
        self.client: Any | None = None
        self.device: str | None = None

    def _load_client(self) -> None:
        """Load CLIP only when an uncached image needs embedding."""
        if self.client is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with suppress_transformers_progress():
            self.processor = CLIPProcessor.from_pretrained(self.model)
            clip_client: Any = CLIPModel.from_pretrained(self.model)
        client = clip_client.to(self.device)
        client.eval()
        self.client = client

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed PIL images as normalized CLIP vectors."""
        self._load_client()
        assert self.client is not None
        assert self.device is not None
        assert self.processor is not None
        assert self.torch is not None
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
            cache_filename=self.cache_filename,
            cache_key=self.model,
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
            cache_filename=self.cache_filename,
            cache_key=self.model,
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
    clip_adapter: ImageEmbedder | None = None,
    batch_size: int = 32,
    verbose: int = 1,
    cache_filename: Pathish | None = None,
    cache_key: str | None = None,
) -> np.ndarray:
    """Embed image paths as normalized CLIP vectors, reusing a cache when given."""
    clip = clip_adapter or ClipImageComparisonMethod()
    cache_path = Path(cache_filename) if cache_filename is not None else None
    cache = _load_embedding_cache(cache_path, cache_key)
    cache_entries = cache.entries
    image_metadata = {
        path: (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in image_paths
    }
    vectors_by_path: dict[Path, np.ndarray] = {}
    uncached_paths: list[Path] = []

    for path, (cache_path_key, size, mtime_ns) in image_metadata.items():
        cached_entry = cache_entries.get(cache_path_key)
        if cached_entry is None:
            uncached_paths.append(path)
            continue
        cached_size, cached_mtime_ns, vector = cached_entry
        if (cached_size, cached_mtime_ns) == (size, mtime_ns):
            vectors_by_path[path] = vector
        else:
            uncached_paths.append(path)

    for start in range(0, len(uncached_paths), batch_size):
        batch_paths = uncached_paths[start : start + batch_size]
        batch_images: list[Image.Image] = []

        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB").copy())

        batch_vectors = clip.embed_images(batch_images)
        for path, vector in zip(batch_paths, batch_vectors, strict=True):
            vectors_by_path[path] = vector
            cache_path_key, size, mtime_ns = image_metadata[path]
            cache_entries[cache_path_key] = (size, mtime_ns, vector)
        if verbose >= 2:
            complete_count = min(start + batch_size, len(uncached_paths))
            print(f"embedded {complete_count}/{len(uncached_paths)}")

    if cache_path is not None and uncached_paths:
        _save_embedding_cache(cache_path, cache_key, cache)

    if not image_paths:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack([vectors_by_path[path] for path in image_paths])


def _load_embedding_cache(
    cache_path: Path | None,
    cache_key: str | None,
) -> EmbeddingCache:
    """Load compatible embedding cache entries without disrupting dedupe."""
    if cache_path is None or not cache_path.is_file():
        return EmbeddingCache(entries={}, reviewed_thresholds={})
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            version = int(cache["version"][0])
            if version not in {1, 2, EMBEDDING_CACHE_VERSION} or str(
                cache["cache_key"][0]
            ) != (cache_key or ""):
                return EmbeddingCache(entries={}, reviewed_thresholds={})
            paths = cache["paths"]
            sizes = cache["sizes"]
            mtimes_ns = cache["mtimes_ns"]
            vectors = cache["vectors"]
            if (
                paths.ndim != 1
                or sizes.shape != paths.shape
                or mtimes_ns.shape != paths.shape
                or vectors.ndim != 2
                or len(vectors) != len(paths)
            ):
                return EmbeddingCache(entries={}, reviewed_thresholds={})
            entries = {
                str(path): (int(size), int(mtime_ns), vector)
                for path, size, mtime_ns, vector in zip(
                    paths,
                    sizes,
                    mtimes_ns,
                    vectors,
                    strict=True,
                )
            }
            if version == EMBEDDING_CACHE_VERSION:
                reviewed_paths = cache["reviewed_paths"]
                automatic_thresholds = cache["reviewed_automatic_thresholds"]
                llm_thresholds = cache["reviewed_llm_thresholds"]
                if (
                    reviewed_paths.ndim != 1
                    or automatic_thresholds.shape != reviewed_paths.shape
                    or llm_thresholds.shape != reviewed_paths.shape
                ):
                    return EmbeddingCache(entries={}, reviewed_thresholds={})
                reviewed_thresholds = {
                    str(path): (float(automatic_threshold), float(llm_threshold))
                    for path, automatic_threshold, llm_threshold in zip(
                        reviewed_paths,
                        automatic_thresholds,
                        llm_thresholds,
                        strict=True,
                    )
                    if str(path) in entries
                }
            else:
                reviewed_paths = (
                    {str(path) for path in cache["processed_paths"]}
                    if version == 2
                    else set(entries)
                )
                reviewed_thresholds = {
                    path: (DEFAULT_AUTOMATIC_THRESHOLD, DEFAULT_LLM_THRESHOLD)
                    for path in reviewed_paths
                    if path in entries
                }
            return EmbeddingCache(
                entries=entries,
                reviewed_thresholds=reviewed_thresholds,
                needs_upgrade=version != EMBEDDING_CACHE_VERSION,
            )
    except (EOFError, KeyError, OSError, ValueError):
        return EmbeddingCache(entries={}, reviewed_thresholds={})


def _save_embedding_cache(
    cache_path: Path,
    cache_key: str | None,
    cache: EmbeddingCache,
) -> None:
    """Atomically save compressed embedding vectors and their file fingerprints."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_entries = sorted(cache.entries.items())
    paths = np.array([path for path, _ in ordered_entries])
    sizes = np.array([entry[0] for _, entry in ordered_entries], dtype=np.int64)
    mtimes_ns = np.array([entry[1] for _, entry in ordered_entries], dtype=np.int64)
    vectors = np.vstack([entry[2] for _, entry in ordered_entries])
    reviewed_items = sorted(
        (path, thresholds)
        for path, thresholds in cache.reviewed_thresholds.items()
        if path in cache.entries
    )
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.npz")
    try:
        np.savez_compressed(
            temporary_path,
            version=np.array([EMBEDDING_CACHE_VERSION]),
            cache_key=np.array([cache_key or ""]),
            paths=paths,
            sizes=sizes,
            mtimes_ns=mtimes_ns,
            vectors=vectors,
            reviewed_paths=np.array([path for path, _ in reviewed_items]),
            reviewed_automatic_thresholds=np.array(
                [thresholds[0] for _, thresholds in reviewed_items],
                dtype=np.float64,
            ),
            reviewed_llm_thresholds=np.array(
                [thresholds[1] for _, thresholds in reviewed_items],
                dtype=np.float64,
            ),
        )
        temporary_path.replace(cache_path)
        cache.needs_upgrade = False
    finally:
        temporary_path.unlink(missing_ok=True)


def _reviewed_cached_image_paths(
    image_paths: Iterable[Path],
    cache_path: Path,
    cache_key: str,
    automatic_threshold: float,
    llm_threshold: float,
) -> set[Path]:
    """Return current images reviewed at compatible dedupe thresholds."""
    cache = _load_embedding_cache(cache_path, cache_key)
    reviewed_paths: set[Path] = set()
    for path in image_paths:
        path_key = str(path.resolve())
        cached_entry = cache.entries.get(path_key)
        reviewed_thresholds = cache.reviewed_thresholds.get(path_key)
        if cached_entry is None or reviewed_thresholds is None:
            continue
        size, mtime_ns, _ = cached_entry
        stat = path.stat()
        if (
            (size, mtime_ns) == (stat.st_size, stat.st_mtime_ns)
            and automatic_threshold >= reviewed_thresholds[0]
            and llm_threshold >= reviewed_thresholds[1]
        ):
            reviewed_paths.add(path)
    return reviewed_paths


def _mark_images_deduped(
    image_paths: Iterable[Path],
    cache_path: Path,
    cache_key: str,
    automatic_threshold: float,
    llm_threshold: float,
) -> None:
    """Record dedupe thresholds for unchanged cached images."""
    cache = _load_embedding_cache(cache_path, cache_key)
    changed = cache.needs_upgrade
    for path in image_paths:
        path_key = str(path.resolve())
        cached_entry = cache.entries.get(path_key)
        if cached_entry is None:
            continue
        size, mtime_ns, _ = cached_entry
        stat = path.stat()
        thresholds = (automatic_threshold, llm_threshold)
        if (size, mtime_ns) == (
            stat.st_size,
            stat.st_mtime_ns,
        ) and cache.reviewed_thresholds.get(path_key) != thresholds:
            cache.reviewed_thresholds[path_key] = thresholds
            changed = True
    if changed:
        _save_embedding_cache(cache_path, cache_key, cache)


def prune_embedding_cache(cache_path: Pathish) -> int:
    """Remove cache entries whose image paths no longer exist."""
    path = Path(cache_path)
    if not path.is_file():
        return 0
    cache = _load_embedding_cache(path, CLIP_MODEL)
    live_entries = {
        path_key: entry
        for path_key, entry in cache.entries.items()
        if Path(path_key).is_file()
    }
    removed_count = len(cache.entries) - len(live_entries)
    if removed_count:
        cache.entries = live_entries
        cache.reviewed_thresholds = {
            path_key: thresholds
            for path_key, thresholds in cache.reviewed_thresholds.items()
            if path_key in live_entries
        }
        _save_embedding_cache(path, CLIP_MODEL, cache)
    return removed_count


def image_dimensions(image_path: Pathish) -> tuple[int, int]:
    """Return image width and height."""
    with Image.open(image_path) as image:
        return image.size


def automatic_duplicate_survivor(
    left_path: Path, right_path: Path
) -> tuple[Path, Path]:
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
        image_base64 = [
            base64_encode_image(left_image),
            base64_encode_image(right_image),
        ]
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
    automatic_threshold: float = DEFAULT_AUTOMATIC_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    batch_size: int = 32,
    comparison_method: ImageComparisonMethod | None = None,
    client_adapter: VisionModelClientAdapter | None = None,
    rejected_llm_matches: list[ImageDuplicateMatch] | None = None,
    verbose: int = 1,
) -> list[ImageDuplicateMatch]:
    """Return accepted duplicate matches from image similarity scores."""
    left_paths = sorted(Path(path) for path in left_images)
    right_paths = (
        None if right_images is None else sorted(Path(path) for path in right_images)
    )
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

    return _dedupe_similarities(
        similarities,
        automatic_threshold=automatic_threshold,
        llm_threshold=llm_threshold,
        provider=provider,
        client_adapter=client_adapter,
        rejected_llm_matches=rejected_llm_matches,
        verbose=verbose,
    )


def dedupe_new_image_matches(
    new_images: Iterable[Pathish],
    processed_images: Iterable[Pathish],
    *,
    automatic_threshold: float = DEFAULT_AUTOMATIC_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    batch_size: int = 32,
    comparison_method: ImageComparisonMethod | None = None,
    client_adapter: VisionModelClientAdapter | None = None,
    rejected_llm_matches: list[ImageDuplicateMatch] | None = None,
    verbose: int = 1,
) -> list[ImageDuplicateMatch]:
    """Find duplicates involving only images not yet processed by dedupe."""
    new_paths = sorted(Path(path) for path in new_images)
    processed_paths = sorted(Path(path) for path in processed_images)
    if not new_paths:
        return []
    method = comparison_method or ClipImageComparisonMethod()
    similarities = method.compare(
        new_paths,
        batch_size=batch_size,
        verbose=verbose,
    )
    if processed_paths:
        similarities.extend(
            method.compare(
                new_paths,
                processed_paths,
                batch_size=batch_size,
                verbose=verbose,
            )
        )
    return _dedupe_similarities(
        similarities,
        automatic_threshold=automatic_threshold,
        llm_threshold=llm_threshold,
        provider=provider,
        client_adapter=client_adapter,
        rejected_llm_matches=rejected_llm_matches,
        verbose=verbose,
    )


def _dedupe_similarities(
    similarities: Iterable[ImageSimilarity],
    *,
    automatic_threshold: float,
    llm_threshold: float,
    provider: VisionModelProvider,
    client_adapter: VisionModelClientAdapter | None,
    rejected_llm_matches: list[ImageDuplicateMatch] | None,
    verbose: int,
) -> list[ImageDuplicateMatch]:
    """Apply automatic and LLM duplicate decisions to similarity candidates."""
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

    owned_adapter = client_adapter is None
    try:
        for similarity in borderline_similarities:
            if {similarity.left_path, similarity.right_path} & planned_removals:
                continue
            if verbose >= 2:
                print(
                    f"LLM duplicate candidate {similarity.score * 100:0.2f}%:\n"
                    f"  left: {format_image_detail(similarity.left_path)}\n"
                    f"  right: {format_image_detail(similarity.right_path)}"
                )
            if client_adapter is None:
                client_adapter = get_vision_model_client_adapter(provider)
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
            kept_path = (
                similarity.left_path
                if judgement.keep == "left"
                else similarity.right_path
            )
            duplicate_path = (
                similarity.right_path
                if judgement.keep == "left"
                else similarity.left_path
            )
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
    finally:
        if owned_adapter and client_adapter is not None:
            client_adapter.cleanup()

    return sorted(matches, key=lambda match: match.score, reverse=True)
