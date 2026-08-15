from pathlib import Path

DEFAULT_WALL_RANDOM_SEED: int = 42

METADATA_FILENAME: Path = Path("image_metadata.csv")
GALLERY_NAME: Path = Path("index.html")

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
