from pathlib import Path

DEFAULT_WALL_RANDOM_SEED: int = 42

METADATA_FILENAME: Path = Path("image_metadata.csv")
GALLERY_NAME: Path = Path("index.html")
DEDUPE_REVIEW_FILENAME: Path = Path("dedupe_review.html")
DEDUPE_EMBEDDINGS_FILENAME: Path = Path("vectors.npz")

IMAGE_EXTENSIONS: list[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".avif",
    ".heic",
]
UNWELCOME_EXTENSIONS: list[str] = [
    ".webp",
    ".avif",
    ".heic",
    ".bmp",
    ".gif",
]
WELCOME_EXTENSIONS: list[str] = [
    extension for extension in IMAGE_EXTENSIONS if extension not in UNWELCOME_EXTENSIONS
]

EXTENSION_IMAGE_FORMATS: dict[str, set[str]] = {
    ".avif": {"AVIF"},
    ".bmp": {"BMP"},
    ".gif": {"GIF"},
    ".heic": {"HEIF"},
    ".jpeg": {"JPEG"},
    ".jpg": {"JPEG"},
    ".png": {"PNG"},
    ".tiff": {"TIFF"},
    ".webp": {"WEBP"},
}

IMAGE_FORMAT_EXTENSIONS: dict[str, str] = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "HEIF": ".heic",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}

LOSSLESS_OR_UNCOMPRESSED_FORMATS: set[str] = {
    "BMP",
    "GIF",
    "PNG",
    "TIFF",
}

LOSSY_FORMATS: set[str] = {
    "AVIF",
    "HEIF",
    "JPEG",
    "WEBP",
}

CLIP_MODEL: str = "openai/clip-vit-base-patch32"
EMBEDDING_CACHE_VERSION: int = 3
DEFAULT_AUTOMATIC_THRESHOLD: float = 0.99
DEFAULT_LLM_THRESHOLD: float = 0.85

DEFAULT_LARGE_IMAGE_THRESHOLD: int = 1_000_000

OPENAI_MODEL: str = "gpt-5.6-luna"
GEMMA_MODEL: str = "gemma4:12b"
QWEN_MODEL: str = "qwen3.5:4b"

WALL_DOUBLE_WIDE_THRESHOLD: float = 1.7
WALL_LAYOUT_MAX_PASSES: int = 3

REVIEW_ID_COLUMN: str = "_review_id"

CSV_COLUMNS: list[str] = [
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
    REVIEW_ID_COLUMN,
]

WALL_TITLE_TEMPLATE: str = """{clean_filename} ({width}x{height})
Category: {category}
Genre: {genre}
Tags: {tags}

{description}
"""
