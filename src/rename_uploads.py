import image_tagger as it
from pathlib import Path

UPLOAD_DIR = Path(r"C:\Users\oloon\Dropbox\images\uploads")
METADATA_FILENAME = UPLOAD_DIR / "image_metadata.csv"


def main(metadata_filename, dry_run=False):
    it.autorename(metadata_filename, verbose=2, dry_run=dry_run)


if __name__ == "__main__":
    main(METADATA_FILENAME, dry_run=False)
