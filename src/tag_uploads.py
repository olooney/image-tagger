import image_tagger as it
from pathlib import Path

UPLOAD_DIR = Path(r"C:\Users\oloon\Dropbox\images\uploads")
METADATA_FILENAME = UPLOAD_DIR / "image_metadata.csv"


def main():
    filepaths = it.find_images(str(UPLOAD_DIR))
    print("number of image files:", len(filepaths))
    it.tag_images(filepaths, str(METADATA_FILENAME), verbose=2)


if __name__ == "__main__":
    main()
