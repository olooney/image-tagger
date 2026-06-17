from pathlib import Path

UPLOAD_DIR = Path(r"C:\Users\oloon\Dropbox\images\uploads")
METADATA_FILENAME = UPLOAD_DIR / "image_metadata.csv"
GALLERY_NAME = UPLOAD_DIR / "index.html"

IMAGE_EXTENSIONS = [
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
UNWELCOME_EXTENSIONS = [".webp", ".avif", ".heic", ".bmp", ".gif"]
WELCOME_EXTENSIONS = [
    extension for extension in IMAGE_EXTENSIONS if extension not in UNWELCOME_EXTENSIONS
]
