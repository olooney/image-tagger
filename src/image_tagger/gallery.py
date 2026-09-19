import os
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, cast

import jinja2
import pandas as pd

from .util import Pathish


def generate_gallery(
    csv_filename: Pathish,
    output_filename: Pathish,
    verbose: int = 1,
) -> None:
    """Generate a static gallery HTML file."""
    csv_path = Path(csv_filename)
    output_path = Path(output_filename)
    metadata_df = pd.read_csv(csv_path)
    metadata_df = metadata_df[metadata_df["status"] == "ok"]
    items: list[dict[str, Any]] = []
    for raw_item in metadata_df.to_dict("records"):
        item = cast("dict[str, Any]", raw_item)
        original_path = Path(item["original_filepath"])
        clean_filename = item.get("clean_filename", "")
        clean_path = (
            original_path.with_name(clean_filename)
            if clean_filename and isinstance(clean_filename, str)
            else None
        )
        if clean_path is not None and clean_path.is_file():
            image_path = clean_path
        elif original_path.is_file():
            image_path = original_path
        else:
            continue

        image_src = os.path.relpath(image_path, output_path.parent)
        item["image_src"] = Path(image_src).as_posix()
        items.append(item)

    first_item = items[0] if items else {}
    provider_name = str(first_item.get("provider_name", "")).strip()
    model = str(first_item.get("model", "")).strip()
    for item in items:
        item["formatted_timestamp"] = datetime.fromisoformat(
            item["timestamp"]
        ).strftime("%m/%d/%y %I:%M %p")
        item["tags"] = [tag.strip() for tag in item["tags"].split(";")]
        notes = item.get("notes", "")
        item["notes"] = notes if notes and isinstance(notes, str) else ""

    template_text = (
        resources.files("image_tagger.data").joinpath("template.html").read_text()
    )
    template = jinja2.Template(template_text)
    output = template.render(items=items, provider_name=provider_name, model=model)

    output_path.write_text(output, encoding="utf-8")
