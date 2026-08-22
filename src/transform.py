"""Perspective correction of flat rectangular images with vision-model guidance."""

import base64
import json
from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal, cast

import cv2
import jinja2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

import image_tagger as it
from constants import WELCOME_EXTENSIONS
from util import quote_display_path


TRANSFORM_REVIEW_FILENAME: Path = Path("transform_review.html")
DEFAULT_MAX_ATTEMPTS: int = 2
Corner = Annotated[list[float], Field(min_length=2, max_length=2)]
RgbColor = Annotated[list[int], Field(min_length=3, max_length=3)]


TransformStrategy = Literal[
    "contour_quadrilateral",
    "line_intersection",
    "no_transform_needed",
]
ReviewDecision = Literal[
    "good_enough",
    "adjust_corners",
    "no_transform_needed",
    "abandon",
]


class BackgroundHint(BaseModel):
    """Approximate background appearance used to suppress irrelevant edges."""

    color_rgb: RgbColor
    confidence: float = Field(ge=0, le=1)
    uniformity: Literal["uniform", "mostly_uniform", "varied"]


class TransformParameters(BaseModel):
    """Fixed detector hints that can be expanded without open-ended JSON."""

    edge_strength: Literal["low", "medium", "high"] | None
    expected_boundary_completeness: float | None = Field(ge=0, le=1)
    notes: str | None


class TransformAssessment(BaseModel):
    """Initial structured routing response from the vision model."""

    strategy: TransformStrategy
    confidence: float = Field(ge=0, le=1)
    reason: str
    approximate_corners: list[Corner] | None = Field(
        min_length=4,
        max_length=4,
    )
    background: BackgroundHint | None
    parameters: TransformParameters


class TransformReview(BaseModel):
    """Structured review response for a proposed perspective crop."""

    decision: ReviewDecision
    confidence: float = Field(ge=0, le=1)
    reason: str
    replacement_corners: list[Corner] | None = Field(
        min_length=4,
        max_length=4,
    )


@dataclass(frozen=True)
class TransformReviewEntry:
    """One source image and the result shown in the static review page."""

    source_path: Path
    source_image_src: str | None
    overlay_image_src: str | None
    transformed_image_src: str | None
    status: str
    reason: str
    attempts: int
    strategy: TransformStrategy
    parameters_json: str
    background_json: str
    review_decision: ReviewDecision | None


class VisionConversation:
    """Keep structured image-transform turns together as one model transcript."""

    def __init__(self, client: it.VisionModelClientAdapter, verbose: int) -> None:
        """Create an empty transcript for one image."""
        self.client = client
        self.verbose = verbose
        self.turns: list[dict[str, object]] = []

    def ask(
        self,
        images: list[Image.Image],
        prompt: str,
        response_format: type[BaseModel],
        label: str,
    ) -> BaseModel:
        """Send a structured turn with its preceding transcript as context."""
        history = ""
        if self.turns:
            history = "\n\nPrevious structured turns in this same image session:\n" + json.dumps(
                self.turns,
                indent=2,
            )
        full_prompt = f"{prompt}{history}"
        encoded_images = [it.base64_encode_image(image) for image in images]
        if self.verbose >= 3:
            print(f"{label} request JSON:")
            print(json.dumps({"prompt": full_prompt}, indent=2))
        result = self.client.vision_task(encoded_images, full_prompt, response_format)
        data = response_format.model_validate(result.data.model_dump())
        response_json = data.model_dump(mode="json")
        if self.verbose >= 3:
            print(f"{label} response JSON:")
            print(json.dumps(response_json, indent=2))
        self.turns.append({"label": label, "prompt": prompt, "response": response_json})
        return data


ASSESSMENT_PROMPT: str = """You are selecting a perspective-correction strategy for a flat rectangular subject.

First decide whether perspective correction is appropriate. Only choose a transform strategy
when one single, dominant, planar rectangular subject is the intended output and it occupies
most of the image. It must need a material improvement from cropping, rotation, or perspective
correction. Choose `no_transform_needed` for a subject that is already nearly front-facing,
closely cropped, and needs only minor cleanup. Also choose `no_transform_needed` when the
image is a broader scene that happens to include a rectangle, such as a box beside a toy, or
when cropping to a rectangle would discard another main subject.

For books, trace only the front cover. Do not include the spine, page block, hand, or nearby
objects in the quadrilateral. Choose `contour_quadrilateral` when the full front-cover boundary
is visible and distinct enough to form a large closed contour. Choose `line_intersection` when
the intended rectangle remains clear but an edge is broken, weak, damaged, obscured, or cut off
by the image frame. In that case, extrapolate the visible supporting edge lines. If their true
intersection lies outside the source image, provide that out-of-bounds corner coordinate; never
substitute an arbitrary in-bounds point or snap it to the image boundary.

Provide four approximate pixel corners in top-left, top-right, bottom-right, bottom-left order
for every transform. Corners may be outside the source frame. If any corner is outside the
frame, provide a background RGB color that represents the surrounding canvas so exposed warped
areas can be filled naturally. Use null for unavailable nonessential hints. Return only the
requested structured response."""

REVIEW_PROMPT: str = """Review a proposed perspective correction. The supplied images are,
in order: the original, the original annotated with a bright green edge-only quadrilateral,
and the perspective-corrected crop.

Choose `good_enough` when the green boundary follows the intended dominant rectangular subject
and the crop materially improves its front-facing presentation. Choose `adjust_corners` when
revised source-image pixel corners would materially improve it; give four replacements in
top-left, top-right, bottom-right, bottom-left order. When a true corner is cut off, extrapolate
the side lines and give the out-of-bounds coordinate rather than snapping to the frame. For a
book, the quadrilateral must follow only the front cover, not the spine or page block. Use null
for replacement_corners otherwise. Choose `no_transform_needed` if the original was already
orthorectified and closely cropped, or if the crop merely isolates a rectangle inside a broader
scene. Choose `abandon` if a credible correction cannot be produced. Return only the requested
structured response."""


def transform_images(
    directory: Path,
    *,
    provider: it.VisionModelProvider = it.VisionModelProvider.OPENAI,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    verbose: int = 1,
    dry_run: bool = False,
    client_adapter: it.VisionModelClientAdapter | None = None,
) -> list[TransformReviewEntry]:
    """Create reviewed perspective-corrected sibling images for a directory."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
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
                    max_attempts=max_attempts,
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
    max_attempts: int,
    verbose: int,
    dry_run: bool,
) -> TransformReviewEntry:
    """Assess, crop, and review one image until accepted or rejected."""
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")
    conversation = VisionConversation(client, verbose)
    assessment = cast(
        TransformAssessment,
        conversation.ask([source], ASSESSMENT_PROMPT, TransformAssessment, "assessment"),
    )

    def entry(
        overlay: Image.Image | None,
        crop: Image.Image | None,
        status: str,
        reason: str,
        attempts: int,
        review: TransformReview | None = None,
    ) -> TransformReviewEntry:
        """Create one report entry using the image's assessment context."""
        return _entry(
            source_path,
            source,
            overlay,
            crop,
            status,
            reason,
            attempts,
            assessment=assessment,
            review=review,
        )

    if assessment.strategy == "no_transform_needed":
        return entry(None, None, "No transform needed", assessment.reason, 0)

    corners_hint = assessment.approximate_corners
    last_overlay: Image.Image | None = None
    for attempt in range(1, max_attempts + 1):
        corners = _corners_for_attempt(source, assessment, corners_hint)
        if corners is None:
            return entry(None, None, "Abandoned", "Could not detect four plausible corners.", attempt)
        overlay = _draw_quadrilateral(source, corners)
        crop = _perspective_crop(source, corners, background=assessment.background)
        if crop is None:
            return entry(overlay, None, "Abandoned", "Detected corners did not form a valid quadrilateral.", attempt)
        last_overlay = overlay
        review = cast(
            TransformReview,
            conversation.ask([source, overlay, crop], REVIEW_PROMPT, TransformReview, "review"),
        )
        if review.decision == "good_enough":
            target_path = _transformed_path(source_path)
            if verbose >= 1:
                print(f"transforming {quote_display_path(source_path)} ...", end="")
            try:
                if not dry_run:
                    crop.save(target_path)
                if verbose >= 1:
                    print("success!")
            except OSError:
                if verbose >= 1:
                    print("error!")
                return entry(overlay, crop, "Write failed", "Could not save the transformed image.", attempt, review)
            status = "Transformed" if not dry_run else "Would transform"
            return entry(overlay, crop, status, review.reason, attempt, review)
        if review.decision == "no_transform_needed":
            return entry(overlay, None, "No transform needed", review.reason, attempt, review)
        if review.decision == "abandon":
            return entry(overlay, crop, "Abandoned", review.reason, attempt, review)
        corners_hint = review.replacement_corners
        if corners_hint is None:
            return entry(overlay, crop, "Abandoned", "Review requested adjustment without replacement corners.", attempt, review)
    return entry(last_overlay, None, "Abandoned", "Maximum attempts reached without approval.", max_attempts)


def _corners_for_attempt(
    source: Image.Image,
    assessment: TransformAssessment,
    corners_hint: list[Corner] | None,
) -> np.ndarray | None:
    """Prefer useful model corners, otherwise run the selected OpenCV detector."""
    if corners_hint is not None:
        return _validate_corners(np.asarray(corners_hint, dtype=np.float32), source.size)
    image = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
    if assessment.strategy == "contour_quadrilateral":
        return _detect_contour_corners(image, assessment.background)
    return _detect_line_corners(image)


def _detect_contour_corners(image: np.ndarray, background: BackgroundHint | None) -> np.ndarray | None:
    """Find a large four-sided external contour."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if background is not None and background.uniformity != "varied":
        background_bgr = np.array(background.color_rgb[::-1], dtype=np.uint8)
        distance = np.linalg.norm(image.astype(np.int16) - background_bgr.astype(np.int16), axis=2)
        edges = cv2.bitwise_or(edges, (distance > 35).astype(np.uint8) * 255)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approximation) == 4:
            return _validate_corners(approximation.reshape(4, 2).astype(np.float32), (image.shape[1], image.shape[0]))
    return None


def _detect_line_corners(image: np.ndarray) -> np.ndarray | None:
    """Fit four broad Hough side lines and intersect their extreme representatives."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=min(image.shape[:2]) // 4, maxLineGap=30)
    if lines is None:
        return None
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    for raw_line in lines[:, 0]:
        x1, y1, x2, y2 = raw_line.astype(float)
        if abs(x2 - x1) >= abs(y2 - y1):
            horizontal.append(raw_line.astype(float))
        else:
            vertical.append(raw_line.astype(float))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
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


def _transformed_path(source_path: Path) -> Path:
    """Return the side-by-side filename for a transformed source image."""
    return source_path.with_name(f"{source_path.stem}.transformed{source_path.suffix}")


def _entry(
    source_path: Path,
    source: Image.Image,
    overlay: Image.Image | None,
    crop: Image.Image | None,
    status: str,
    reason: str,
    attempts: int,
    *,
    assessment: TransformAssessment,
    review: TransformReview | None,
) -> TransformReviewEntry:
    """Create a report entry with self-contained image data URLs."""
    return TransformReviewEntry(
        source_path,
        _data_url(source),
        _data_url(overlay),
        _data_url(crop),
        status,
        reason,
        attempts,
        assessment.strategy,
        assessment.parameters.model_dump_json(indent=2),
        json.dumps(
            assessment.background.model_dump(mode="json")
            if assessment.background is not None
            else None,
            indent=2,
        ),
        review.decision if review is not None else None,
    )


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