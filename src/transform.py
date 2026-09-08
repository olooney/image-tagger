"""Perspective correction of flat rectangular images with vision-model guidance."""

import base64
import json
import time
from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

import cv2
import jinja2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from send2trash import send2trash

import image_tagger as it
from constants import WELCOME_EXTENSIONS
from util import quote_display_path


TRANSFORM_REVIEW_FILENAME: Path = Path("transform_review.html")
Corner = Annotated[list[float], Field(min_length=2, max_length=2)]
RgbColor = Annotated[list[int], Field(min_length=3, max_length=3)]
RelativeCoordinate = Annotated[float, Field(allow_inf_nan=False)]
NormalizedCorner = Annotated[
    list[RelativeCoordinate],
    Field(min_length=2, max_length=2),
]


class BackgroundHint(BaseModel):
    """Approximate background appearance used to suppress irrelevant edges."""

    color_rgb: RgbColor
    confidence: float = Field(ge=0, le=1)
    uniformity: Literal["uniform", "mostly_uniform", "varied"]


class CropVlmDecision(BaseModel):
    """Structured VLM choice, reasoning, and normalized crop corners."""

    which: Literal["contour", "hough", "neither"]
    adjustments: str
    points: list[NormalizedCorner] = Field(min_length=4, max_length=4)


@dataclass(frozen=True)
class CropDetection:
    """CV candidates and the VLM-refined crop for one source image."""

    size: tuple[int, int]
    background: BackgroundHint
    contour_corners: np.ndarray
    hough_corners: np.ndarray
    final_corners: np.ndarray
    which: Literal["contour", "hough", "neither"]
    adjustments: str


@dataclass(frozen=True)
class TransformReviewEntry:
    """Four debug views and the crop result for one source image."""

    source_path: Path
    hough_image_src: str
    contour_image_src: str
    vlm_image_src: str
    transformed_image_src: str | None
    status: str
    which: Literal["contour", "hough", "neither"]
    adjustments: str


CROP_VLM_PROMPT: str = """Choose and refine the crop for the front cover of the book shown.
You receive three images in this exact order:
Image 1: original.
Image 2: contour guess overlay.
Image 3: Hough guess overlay.

The contour and Hough guesses are also provided as JSON below. Coordinates are normalized x/y
values where 0 to 1 is the range of the source image, ordered top-left, top-right, bottom-right,
bottom-left. Coordinates may go somewhat outside that range when a true corner is beyond the
image frame. For example, (-0.2, 0.3) is perfectly valid and must not be clamped to the image.
{guesses_json}

First choose which guess, if either, best fits the front cover. Then explain what adjustments
you would make. Finally return exactly four refined points in the same order and coordinate
system. If unsure, or if the image already appears correctly cropped, use the full-image
coordinates [[0, 0], [1, 0], [1, 1], [0, 1]]."""


def transform_images(
    directory: Path,
    *,
    provider: it.VisionModelProvider = it.VisionModelProvider.OPENAI,
    verbose: int = 1,
    dry_run: bool = False,
    client_adapter: it.VisionModelClientAdapter | None = None,
) -> list[TransformReviewEntry]:
    """Detect and apply one VLM-refined perspective crop per image."""
    directory = Path(directory)
    image_paths = [
        path
        for path in sorted(it.find_images(directory, extension_filter=WELCOME_EXTENSIONS))
        if ".transformed." not in path.name.lower()
    ]
    if verbose == 1:
        print(f"working in {quote_display_path(directory)}")
    owned_adapter = client_adapter is None
    client = client_adapter or it.get_vision_model_client_adapter(provider)
    entries: list[TransformReviewEntry] = []
    try:
        for source_path in image_paths:
            entries.append(
                _transform_one_image(
                    source_path,
                    client=client,
                    verbose=verbose,
                    dry_run=dry_run,
                )
            )
    finally:
        if owned_adapter:
            client.cleanup()
    review_filename = directory / TRANSFORM_REVIEW_FILENAME
    generate_transform_review(entries, review_filename)
    if verbose >= 1:
        print(f"wrote {quote_display_path(review_filename)}")
    return entries


def _transform_one_image(
    source_path: Path,
    *,
    client: it.VisionModelClientAdapter,
    verbose: int,
    dry_run: bool,
) -> TransformReviewEntry:
    """Detect, report, and optionally apply one perspective crop."""
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")
    detection = detect_crop(source, client=client, verbose=verbose)
    hough_overlay = _draw_quadrilateral(source, detection.hough_corners)
    contour_overlay = _draw_quadrilateral(source, detection.contour_corners)
    vlm_overlay = _draw_quadrilateral(source, detection.final_corners)
    is_full = np.allclose(detection.final_corners, _full_corners(source.size), atol=1)
    crop = source.copy() if is_full else _perspective_crop(
        source,
        detection.final_corners,
        background=detection.background,
    )
    if crop is None:
        status = "Invalid crop"
    elif is_full:
        status = "No transform needed"
    else:
        status = "Would transform" if dry_run else "Transformed"
        if verbose >= 1:
            print(f"transforming {quote_display_path(source_path)} ...", end="")
        temporary_path = source_path.with_name(
            f".{source_path.stem}.crop-{time.time_ns()}{source_path.suffix}"
        )
        try:
            if not dry_run:
                crop.save(temporary_path)
                send2trash(source_path)
                temporary_path.replace(source_path)
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            status = "Write failed"
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return TransformReviewEntry(
        source_path=source_path,
        hough_image_src=_data_url(hough_overlay),
        contour_image_src=_data_url(contour_overlay),
        vlm_image_src=_data_url(vlm_overlay),
        transformed_image_src=_data_url(crop),
        status=status,
        which=detection.which,
        adjustments=detection.adjustments,
    )


def _full_corners(size: tuple[int, int]) -> np.ndarray:
    """Return source-image corner pixels in clockwise order."""
    width, height = size
    return np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def _background_hint(source: Image.Image) -> BackgroundHint:
    """Summarize border color and variation for crop detection and fill."""
    source_array = np.asarray(source)
    edge_pixels = np.concatenate(
        [source_array[0], source_array[-1], source_array[:, 0], source_array[:, -1]],
    )
    color = np.median(edge_pixels, axis=0).astype(np.uint8)
    distances = np.linalg.norm(edge_pixels.astype(np.int16) - color.astype(np.int16), axis=1)
    spread = float(np.percentile(distances, 90))
    uniformity: Literal["uniform", "mostly_uniform", "varied"]
    if spread <= 12:
        uniformity = "uniform"
    elif spread <= 35:
        uniformity = "mostly_uniform"
    else:
        uniformity = "varied"
    return BackgroundHint(
        color_rgb=color.tolist(),
        confidence=float(np.mean(distances <= 35)),
        uniformity=uniformity,
    )


def detect_crop(
    source: Image.Image,
    *,
    client: it.VisionModelClientAdapter,
    verbose: int = 1,
) -> CropDetection:
    """Refine contour and Hough crop guesses with one structured VLM call."""
    source = source.convert("RGB")
    width, height = source.size
    background = _background_hint(source)
    image = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
    full_corners = _full_corners(source.size)
    contour_corners = _detect_contour_corners(image, background)
    hough_corners = _detect_line_corners(image, background)
    contour_corners = full_corners if contour_corners is None else contour_corners
    hough_corners = full_corners if hough_corners is None else hough_corners

    def normalize(corners: np.ndarray) -> list[list[float]]:
        """Scale pixel corners into source-relative coordinates."""
        return [
            [float(x) / max(1, width - 1), float(y) / max(1, height - 1)]
            for x, y in corners
        ]

    guesses = {
        "contour": normalize(contour_corners),
        "hough": normalize(hough_corners),
    }
    vlm_images = [
        source,
        _draw_quadrilateral(source, contour_corners),
        _draw_quadrilateral(source, hough_corners),
    ]
    if max(source.size) > 1200:
        vlm_images = [
            it.resize_image_to_fit(image, max_dimension=1000)
            for image in vlm_images
        ]
    result = client.vision_task(
        [it.base64_encode_image(image) for image in vlm_images],
        CROP_VLM_PROMPT.format(guesses_json=json.dumps(guesses, indent=2)),
        CropVlmDecision,
    )
    decision = CropVlmDecision.model_validate(result.data.model_dump())
    if verbose >= 2:
        print(f"VLM selected {decision.which}.")
        print(f"VLM adjustments: {decision.adjustments}")
    final_corners = np.asarray(
        [[x * (width - 1), y * (height - 1)] for x, y in decision.points],
        dtype=np.float32,
    )
    return CropDetection(
        size=source.size,
        background=background,
        contour_corners=contour_corners,
        hough_corners=hough_corners,
        final_corners=final_corners,
        which=decision.which,
        adjustments=decision.adjustments,
    )


def detect_quad(
    source_path: Path,
    client_adapter: it.VisionModelClientAdapter,
    verbose: int = 1,
) -> list[list[float]]:
    """Return VLM-refined source-relative corners for an image."""
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")
    detection = detect_crop(source, client=client_adapter, verbose=verbose)
    width, height = source.size
    return [
        [float(x) / max(1, width - 1), float(y) / max(1, height - 1)]
        for x, y in detection.final_corners
    ]


def _detection_edges(image: np.ndarray, background: BackgroundHint | None) -> np.ndarray:
    """Find broad luminance edges, supplemented by a stable background color."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    if background is not None and background.uniformity != "varied":
        background_bgr = np.array(background.color_rgb[::-1], dtype=np.uint8)
        distance = np.linalg.norm(image.astype(np.int16) - background_bgr.astype(np.int16), axis=2)
        foreground = (distance > 35).astype(np.uint8) * 255
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        edges = cv2.bitwise_or(edges, cv2.morphologyEx(foreground, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
    return edges


def _detect_contour_corners(image: np.ndarray, background: BackgroundHint | None) -> np.ndarray | None:
    """Find a large four-sided external contour."""
    edges = _detection_edges(image, background)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approximation) == 4:
            return _validate_corners(approximation.reshape(4, 2).astype(np.float32), (image.shape[1], image.shape[0]))
    return None


def _detect_line_corners(
    image: np.ndarray,
    background: BackgroundHint | None = None,
) -> np.ndarray | None:
    """Fit four broad Hough side lines and intersect their extreme representatives."""
    edges = _detection_edges(image, background)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=min(image.shape[:2]) // 4, maxLineGap=30)
    if lines is None:
        return _detect_contour_corners(image, background)
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    for raw_line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = raw_line.astype(float)
        if abs(x2 - x1) >= abs(y2 - y1):
            horizontal.append(raw_line.astype(float))
        else:
            vertical.append(raw_line.astype(float))
    if len(horizontal) < 2 or len(vertical) < 2:
        return _detect_contour_corners(image, background)
    top, bottom = _extreme_lines(horizontal, axis=1)
    left, right = _extreme_lines(vertical, axis=0)
    corners = np.asarray([
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    ], dtype=np.float32)
    if not np.isfinite(corners).all():
        return None
    return _validate_corners(corners, (image.shape[1], image.shape[0]))


def _extreme_lines(lines: list[np.ndarray], *, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the two side-line candidates furthest apart along one axis."""
    ordered = sorted(lines, key=lambda line: (line[axis] + line[axis + 2]) / 2)
    return ordered[0], ordered[-1]


def _line_intersection(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return the intersection point of two infinite Hough line segments."""
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return float("nan"), float("nan")
    first_cross = x1 * y2 - y1 * x2
    second_cross = x3 * y4 - y3 * x4
    return (
        (first_cross * (x3 - x4) - (x1 - x2) * second_cross) / denominator,
        (first_cross * (y3 - y4) - (y1 - y2) * second_cross) / denominator,
    )


def _validate_corners(corners: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """Order and validate a four-corner quadrilateral, allowing modest overscan."""
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    ordered = _order_corners(corners)
    width, height = size
    if np.any(ordered[:, 0] < -width * 0.5) or np.any(ordered[:, 0] > width * 1.5):
        return None
    if np.any(ordered[:, 1] < -height * 0.5) or np.any(ordered[:, 1] > height * 1.5):
        return None
    if abs(cv2.contourArea(ordered.reshape((-1, 1, 2)))) < width * height * 0.03:
        return None
    return ordered


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).reshape(-1)
    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(differences)]
    ordered[3] = corners[np.argmax(differences)]
    return ordered


def _perspective_crop(
    source: Image.Image,
    corners: np.ndarray,
    *,
    background: BackgroundHint | None = None,
) -> Image.Image | None:
    """Warp a validated quadrilateral into its front-facing rectangular crop."""
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    width = int(max(top, bottom))
    height = int(max(left, right))
    if width < 2 or height < 2:
        return None
    matrix = cv2.getPerspectiveTransform(corners, np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32))
    if background is not None:
        border_color = tuple(background.color_rgb)
        warped = cv2.warpPerspective(
            np.asarray(source),
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color,
        )
    else:
        warped = cv2.warpPerspective(
            np.asarray(source),
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )
    return Image.fromarray(warped)


def _draw_quadrilateral(source: Image.Image, corners: np.ndarray) -> Image.Image:
    """Draw an edge-only green quadrilateral over a source image."""
    overlay = source.copy()
    line_width = max(3, min(source.size) // 200)
    ImageDraw.Draw(overlay).line([*map(tuple, corners), tuple(corners[0])], fill=(0, 255, 0), width=line_width)
    return overlay


def _data_url(image: Image.Image | None) -> str | None:
    """Encode a report image as a bounded PNG data URL."""
    if image is None:
        return None
    thumbnail = image.copy()
    thumbnail.thumbnail((500, 750), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumbnail.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def generate_transform_review(entries: list[TransformReviewEntry], output_filename: Path) -> Path:
    """Render the static transform review report."""
    template_text = resources.files("image_tagger_data").joinpath("transform_review.html").read_text(encoding="utf-8")
    template = jinja2.Environment(autoescape=True).from_string(template_text)
    output_filename.write_text(template.render(entries=entries), encoding="utf-8")
    return output_filename