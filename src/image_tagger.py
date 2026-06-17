import base64
import csv
import json
import os
import random
import re
import string
import subprocess
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import jinja2
import pandas as pd
import requests
from PIL import Image
from pydantic import BaseModel

from constants import WELCOME_EXTENSIONS
from util import Pathish, TemporarySeed, connect_to_openai, make_unique


class ImageTagData(BaseModel):
    """Structured metadata returned by vision models."""

    description: str
    category: str
    genre: str
    tags: list[str]
    filename_already_makes_sense: bool
    filename: str


IMAGE_PROMPT_TEMPLATE: str = (
    resources.files("image_tagger_data").joinpath("image_prompt.md").read_text()
)

csv_columns: list[str] = [
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
]


def quote_display_path(path: Pathish) -> str:
    """Quote a path for command-line display."""
    return subprocess.list2cmdline([os.fspath(path)])


def display_path(
    path: Pathish,
    *,
    verbose: int,
    relative_to: Pathish,
) -> str:
    """Format a path for verbose output."""
    display_path = Path(path)
    if verbose == 1:
        display = display_path.relative_to(relative_to).as_posix()
    else:
        display = os.fspath(display_path)
    return quote_display_path(display)


class VisionModelProvider(Enum):
    """Supported vision model providers."""

    OPENAI = "openai"
    GEMMA = "gemma"
    QWEN = "qwen"


GEMMA_MODEL: str = "gemma4:e4b"
OPENAI_MODEL: str = "gpt-5.4"
QWEN_MODEL: str = "qwen3.5:4b"


@dataclass(frozen=True)
class VisionTaskResult:
    """Raw vision model response data."""

    content: str
    model: str
    total_tokens: int


class VisionModelClientAdapter(ABC):
    """Common interface for vision providers."""

    provider_name: str
    model: str

    def __str__(self) -> str:
        """Format the provider for console output."""
        return f"{self.provider_name} ({self.model})"

    def __repr__(self) -> str:
        """Format the provider for debugging."""
        return f"{self.__class__.__name__}(model={self.model!r})"

    @abstractmethod
    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run a vision task."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release provider resources."""
        pass


class OpenAIVisionModelClientAdapter(VisionModelClientAdapter):
    """OpenAI vision provider adapter."""

    provider_name = "OpenAI"

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        """Create an OpenAI adapter."""
        self.model = model
        self.client = connect_to_openai()

    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run an OpenAI vision request."""
        url = f"data:image/png;base64,{image_base64}"
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            response_format=response_format,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        },
                    ],
                },
            ],
        )
        json_string = response.choices[0].message.content
        if json_string is None:
            raise ValueError("OpenAI response did not include message content.")
        total_tokens = response.usage.total_tokens if response.usage is not None else 0
        return VisionTaskResult(
            content=json_string,
            model=response.model,
            total_tokens=total_tokens,
        )

    def cleanup(self) -> None:
        """OpenAI cleanup hook."""
        pass


class OllamaVisionModelClientAdapter(VisionModelClientAdapter):
    """Ollama vision provider adapter."""

    provider_name = "Ollama"

    def __init__(self, model: str) -> None:
        """Create an Ollama adapter."""
        import ollama

        self.model = model
        self.client = ollama.Client()

    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run an Ollama vision request."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                },
            ],
            format=response_format.model_json_schema(),
            options={
                "temperature": 0,
                "image_min_tokens": 1120,
                "image_max_tokens": 1120,
            },
        )
        message = response.get("message", {})
        return VisionTaskResult(
            content=message.get("content", ""),
            model=response.get("model", self.model),
            total_tokens=response.get("prompt_eval_count", 0)
            + response.get("eval_count", 0),
        )

    def cleanup(self) -> None:
        """Unload the Ollama model."""
        self.client.generate(model=self.model, prompt="", keep_alive=0)


_vision_model_client_adapters: dict[VisionModelProvider, VisionModelClientAdapter] = {}


def get_vision_model_client_adapter(
    provider: VisionModelProvider,
) -> VisionModelClientAdapter:
    """Return a cached adapter for a provider."""
    provider = VisionModelProvider(provider)
    if provider not in _vision_model_client_adapters:
        if provider == VisionModelProvider.OPENAI:
            _vision_model_client_adapters[provider] = OpenAIVisionModelClientAdapter()
        elif provider == VisionModelProvider.GEMMA:
            _vision_model_client_adapters[provider] = OllamaVisionModelClientAdapter(
                GEMMA_MODEL
            )
        elif provider == VisionModelProvider.QWEN:
            _vision_model_client_adapters[provider] = OllamaVisionModelClientAdapter(
                QWEN_MODEL
            )
        else:
            raise ValueError(f"Unsupported vision model provider: {provider.value}")
    return _vision_model_client_adapters[provider]


def clean_filename(filename: str) -> str:
    """Clean up a suggested filename."""
    filename = filename.lower()
    filename = re.sub(r"^[^a-zA-Z_]+", "", filename)  # strip leading whitespace
    filename = re.sub(r"[\s_-]+", "_", filename)  # whitespace to underscore
    filename = re.sub(r"[^a-zA-Z0-9_.]", "", filename)  # strip special characters
    filename = re.sub(r"[\s_-]*\.+", ".", filename)  # whitespace before dot

    return filename


def fix_extension(current_filename: str, suggested_filename: str) -> str:
    """Force a suggested filename to keep its original extension."""
    current_path = Path(current_filename)
    suggested_path = Path(suggested_filename)
    if current_path.suffix.lower() != suggested_path.suffix.lower():
        suggested_path = suggested_path.with_suffix(current_path.suffix)
    return suggested_path.name


def path_name_ext(path: Pathish) -> tuple[str, str, str]:
    """Split a path into directory, stem, and extension."""
    image_path = Path(path)
    return (os.fspath(image_path.parent) + os.sep, image_path.stem, image_path.suffix)


def scramble(filename: str) -> str:
    """Hash a filename to obscure it for testing."""
    with TemporarySeed(seed=hash(filename)):
        return "".join(random.sample(string.ascii_letters, k=8))


def resize_image_to_fit(
    image: Image.Image | Pathish,
    max_dimension: int = 512,
) -> Image.Image:
    """Resize an image to fit inside a square."""
    # read from disk if given a filename
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    original_width, original_height = image.size

    # Determine which dimension is larger and calculate scaling factor
    if max(original_width, original_height) > max_dimension:
        if original_width > original_height:
            scaling_factor = max_dimension / original_width
        else:
            scaling_factor = max_dimension / original_height

        # calculate new dimensions from the chosen scale
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # resize with high-quality downsampling
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


def base64_encode_image(image: Image.Image | Pathish) -> str:
    """Encode an image as base64 PNG data."""

    if not isinstance(image, Image.Image):
        image = Image.open(os.fspath(image))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_data = buffer.getvalue()
    base64_encoded_bytes = base64.b64encode(byte_data)

    return base64_encoded_bytes.decode("utf-8")


def tag_image(
    filepath: Pathish,
    client_adapter: VisionModelClientAdapter,
    prompt_template: str = IMAGE_PROMPT_TEMPLATE,
) -> dict[str, Any]:
    """Tag a single image with a vision model."""
    # handle local or remote images
    filepath_string = os.fspath(filepath)
    if filepath_string.startswith("http"):
        url = filepath_string
        filename = urlsplit(url).path.split("/")[-1]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = resize_image_to_fit(image)
    else:
        image_path = Path(filepath)
        filename = image_path.name
        image = resize_image_to_fit(image_path)

    base64_image_data = base64_encode_image(image)

    # run the tagging vision task and record the time it took
    prompt = prompt_template.format(filename=filename)
    vision_start_time = time.perf_counter()
    response = client_adapter.vision_task(base64_image_data, prompt, ImageTagData)
    vision_duration = time.perf_counter() - vision_start_time
    data = json.loads(response.content)

    # validate the response
    ImageTagData(**data)

    # clean up the suggested filename and fix the extension if necessary
    suggested_filename = clean_filename(data.get("filename", None))
    suggested_filename_fixed = fix_extension(filename, suggested_filename)

    # format the results
    data["clean_filename"] = suggested_filename_fixed
    data["original_filepath"] = filepath_string
    data["original_filename"] = filename
    data["total_tokens"] = response.total_tokens
    data["provider_name"] = client_adapter.provider_name
    data["model"] = response.model
    data["width"] = image.size[0]
    data["height"] = image.size[1]
    data["vision_duration"] = vision_duration

    return data


def tag_images(
    filepaths: Iterable[Pathish],
    output_filename: Pathish,
    retry_errors: bool = False,
    verbose: int = 1,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    instructions_filename: Pathish | None = None,
) -> None:
    """Tag images and write metadata rows."""
    output_path = Path(output_filename)
    client_adapter = get_vision_model_client_adapter(provider)
    if instructions_filename is None:
        prompt_template = IMAGE_PROMPT_TEMPLATE
    else:
        prompt_template = Path(instructions_filename).read_text(encoding="utf-8")
    if verbose >= 1:
        print(f"Using {client_adapter}")
    file_already_exists = output_path.exists()
    mode = "a" if file_already_exists else "w"

    try:
        with output_path.open(mode, newline="", encoding="utf-8") as csvfile:
            columns = csv_columns
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            if not file_already_exists:
                writer.writeheader()

            vision_durations = []

            for index, filepath in enumerate(filepaths):
                row_start_time = time.perf_counter()

                try:
                    # run the model and normalize row fields for CSV output
                    row = tag_image(filepath, client_adapter, prompt_template)
                    duration = row.pop("vision_duration")
                    vision_durations.append(duration)
                    row["tags"] = ";".join(tag.lower().strip() for tag in row["tags"])
                    row.update(
                        {"timestamp": datetime.now().isoformat(), "status": "ok"}
                    )
                    writer.writerow(row)
                    csvfile.flush()

                    if verbose == 0:
                        print(".", end=("\n" if (index + 1) % 100 == 0 else ""))
                    elif verbose == 1:
                        average_durations = (
                            vision_durations[1:]
                            if len(vision_durations) > 2
                            else vision_durations
                        )
                        average_duration = sum(average_durations) / len(
                            average_durations
                        )
                        print(
                            f"{row['timestamp']} {row['original_filename']} -> "
                            f"{row['clean_filename']}: {row['category']} {row['genre']} {row['status']} "
                            f"{duration:0.2f}s avg {average_duration:0.2f}s"
                        )
                    elif verbose >= 2:
                        print(repr(row))
                except KeyboardInterrupt:
                    if verbose >= 1:
                        print("\nInterrupted; cleaning up...")
                    raise
                except Exception:
                    error_message = traceback.format_exc()
                    duration = time.perf_counter() - row_start_time

                    if verbose == 1:
                        print("e", end=("\n" if (index + 1) % 100 == 0 else ""))
                    elif verbose == 2:
                        original_filename = Path(filepath).name
                        print(
                            f"{datetime.now().isoformat()} {original_filename} -> "
                            f"<none> error {duration:0.2f}s"
                        )
                    elif verbose >= 3:
                        print(error_message)

                    writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "original_filepath": filepath,
                            "status": "error",
                            "description": error_message,
                        }
                    )
    finally:
        client_adapter.cleanup()


def previously_tagged_filenames(metadata_filename: Pathish) -> set[str]:
    """Return filenames already tagged successfully."""
    metadata_path = Path(metadata_filename)
    if not metadata_path.exists():
        return set()

    tagged_filenames: set[str] = set()
    with metadata_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("status") != "ok":
                continue

            for column in ["original_filename", "clean_filename"]:
                value = row.get(column)
                if not value:
                    continue
                tagged_filenames.add(Path(value).name)

    return tagged_filenames


def find_images(
    dirs: Pathish | Iterable[Pathish],
    max_days_old: float | None = None,
    metadata_filename: Pathish | None = None,
    extension_filter: Iterable[str] | None = WELCOME_EXTENSIONS,
) -> list[Path]:
    """Find untagged image files in directories."""
    if max_days_old is None:
        max_days_old = float("Inf")

    if isinstance(dirs, (str, os.PathLike)):
        directories = [dirs]
    else:
        directories = dirs

    tagged_filenames = (
        previously_tagged_filenames(metadata_filename)
        if metadata_filename is not None
        else set()
    )
    allowed_extensions = (
        {extension.lower() for extension in extension_filter}
        if extension_filter is not None
        else None
    )
    current_time = time.time()

    filepaths: list[Path] = []
    for directory in directories:
        directory_path = Path(cast("Any", directory))
        for filepath in directory_path.iterdir():
            if not filepath.is_file():
                continue
            if (current_time - filepath.stat().st_mtime) >= max_days_old * 86400:
                continue
            if allowed_extensions is not None:
                if filepath.suffix.lower() not in allowed_extensions:
                    continue
            if filepath.name in tagged_filenames:
                continue
            filepaths.append(filepath)

    return filepaths


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


def rename_images(
    csv_filename: Pathish,
    verbose: int = 1,
    dry_run: bool = False,
) -> None:
    """Rename images from metadata suggestions."""
    csv_path = Path(csv_filename)
    metadata_df = pd.read_csv(csv_path)
    metadata_updated = False
    display_directory = csv_path.parent
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for index, row in metadata_df.iterrows():
        source = Path(row["original_filepath"])

        if row["status"] != "ok" or not row["clean_filename"]:
            if verbose >= 2:
                print(f"skipping errored row {index} {source!r}")
            continue

        # new filename
        new_filename = row["clean_filename"]
        target = source.with_name(new_filename)

        # old filename
        if not source.is_file():
            if verbose >= 2:
                print(f"source file {os.fspath(source)!r} is missing!")
            if verbose >= 1 and not target.is_file():
                print(
                    f"both source file {os.fspath(source)!r} and {os.fspath(target)!r} are missing!"
                )
            continue

        # check for no-op
        if target == source:
            if verbose >= 2:
                print(f"no rename necessary for {source!r}")
            continue

        # ensure extension matches
        if source.suffix.lower() != target.suffix.lower():
            if verbose >= 1:
                print(
                    f"Mismatched file extensions between {os.fspath(source)!r} and {os.fspath(target)!r}; skipping rename!"
                )
            continue

        # check for name collisions
        if target.is_file():
            if verbose >= 1:
                print(f"target {os.fspath(target)!r} already exists!")
            target = Path(make_unique(target))
            if verbose >= 1:
                print(f"proceeding with target {os.fspath(target)!r}.")

        # actually perform the file rename
        if verbose >= 1:
            display_source = display_path(
                source, verbose=verbose, relative_to=display_directory
            )
            display_target = display_path(
                target, verbose=verbose, relative_to=display_directory
            )
            print(f"renaming {display_source} to {display_target} ...", end="")
        try:
            if not dry_run:
                source.rename(target)
                target_filename = target.name
                if row["clean_filename"] != target_filename:
                    metadata_df.at[index, "clean_filename"] = target_filename
                    metadata_updated = True
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            else:
                print(f"error renaming {os.fspath(source)!r} to {os.fspath(target)!r}!")
            traceback.print_exc()

    if metadata_updated:
        metadata_df.to_csv(csv_path, index=False)


def shelve_images(
    csv_filename: Pathish,
    verbose: int = 1,
    dry_run: bool = False,
) -> None:
    """Move images into category folders."""
    csv_path = Path(csv_filename)
    metadata_df = pd.read_csv(csv_path)
    upload_directory = csv_path.parent
    display_directory = upload_directory.parent
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for index, row in metadata_df.iterrows():
        original_source = Path(row["original_filepath"])

        if row["status"] != "ok" or not row["category"]:
            if verbose >= 2:
                print(f"skipping row {index} {original_source!r}")
            continue

        source_filename = row["clean_filename"] or row["original_filename"]
        source = original_source.with_name(source_filename)
        if not source.is_file():
            source = original_source

        if not source.is_file():
            if verbose >= 1:
                print(f"source file {os.fspath(source)!r} is missing!")
            continue

        category = str(row["category"]).strip()
        target_directory = source.parent.parent / category
        if not target_directory.is_dir():
            if verbose >= 1:
                print(
                    f"target directory {os.fspath(target_directory)!r} is missing; skipping {os.fspath(source)!r}"
                )
            continue

        target = target_directory / source.name

        if target == source:
            if verbose >= 2:
                print(f"no move necessary for {source!r}")
            continue

        if target.is_file():
            if verbose >= 1:
                print(f"target {os.fspath(target)!r} already exists!")
            target = Path(make_unique(target))
            if verbose >= 1:
                print(f"proceeding with target {os.fspath(target)!r}.")

        if verbose >= 1:
            display_source = display_path(
                source, verbose=verbose, relative_to=display_directory
            )
            display_target = display_path(
                target, verbose=verbose, relative_to=display_directory
            )
            print(f"moving {display_source} to {display_target} ...", end="")
        try:
            if not dry_run:
                source.rename(target)
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            else:
                print(f"error moving {os.fspath(source)!r} to {os.fspath(target)!r}!")
            traceback.print_exc()


def generate_gallery(
    csv_filename: Pathish,
    output_filename: Pathish,
    verbose: int = 1,
) -> None:
    """Generate a static gallery HTML file."""
    # read the metadata and prepare for merge
    metadata_df = pd.read_csv(Path(csv_filename))
    metadata_df = metadata_df[metadata_df["status"] == "ok"]
    items = metadata_df.to_dict("records")
    first_item = items[0] if items else {}
    provider_name = str(first_item.get("provider_name", "")).strip()
    model = str(first_item.get("model", "")).strip()
    for item in items:
        item["formatted_timestamp"] = datetime.fromisoformat(
            item["timestamp"]
        ).strftime("%m/%d/%y %I:%M %p")
        item["tags"] = [tag.strip() for tag in item["tags"].split(";")]
        notes = item.get("notes", "")
        # filter out the NaNs that pandas uses for missing values.
        item["notes"] = notes if notes and isinstance(notes, str) else ""

    # Render the template with the data
    template_text = (
        resources.files("image_tagger_data").joinpath("template.html").read_text()
    )
    template = jinja2.Template(template_text)
    output = template.render(items=items, provider_name=provider_name, model=model)

    # Save the rendered HTML to a file
    Path(output_filename).write_text(output, encoding="utf-8")
