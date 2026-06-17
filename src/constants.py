from pathlib import Path

UPLOAD_DIR: Path = Path(r"C:\Users\oloon\Dropbox\images\uploads")
METADATA_FILENAME: Path = UPLOAD_DIR / "image_metadata.csv"
GALLERY_NAME: Path = UPLOAD_DIR / "index.html"

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
