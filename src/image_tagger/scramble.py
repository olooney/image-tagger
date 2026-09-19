from pathlib import Path

from .compare import resize_image_to_fit
from .tagging import find_images
from .util import Pathish, scramble


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
