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
from typing import Any
from urllib.parse import urlsplit

import jinja2
import pandas as pd
import requests
from PIL import Image
from pydantic import BaseModel

from constants import WELCOME_EXTENSIONS
from util import Pathish, TemporarySeed, connect_to_openai, make_unique


class ImageTagData(BaseModel):
    description: str
    category: str
    genre: str
    tags: list[str]
    filename_already_makes_sense: bool
    filename: str


IMAGE_PROMPT_TEMPLATE = (
    resources.files("image_tagger_data").joinpath("image_prompt.md").read_text()
)

csv_columns = [
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


def quote_display_path(path: str | os.PathLike[str]) -> str:
    return subprocess.list2cmdline([os.fspath(path)])


def display_path(
    path: str | os.PathLike[str],
    *,
    verbose: int,
    relative_to: str | os.PathLike[str],
) -> str:
    display = os.fspath(path)
    if verbose == 1:
        display = os.path.relpath(display, os.fspath(relative_to))
        display = display.replace(os.sep, "/")
    return quote_display_path(display)


class VisionModelProvider(Enum):
    OPENAI = "openai"
    GEMMA = "gemma"
    QWEN = "qwen"


GEMMA_MODEL = "gemma4:e4b"
OPENAI_MODEL = "gpt-5.4"
QWEN_MODEL = "qwen3.5:4b"


@dataclass(frozen=True)
class VisionTaskResult:
    content: str
    model: str
    total_tokens: int


class VisionModelClientAdapter(ABC):
    provider_name: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider_name} ({self.model})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"

    @abstractmethod
    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class OpenAIVisionModelClientAdapter(VisionModelClientAdapter):
    provider_name = "OpenAI"

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        self.model = model
        self.client = connect_to_openai()

    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
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
        pass


class OllamaVisionModelClientAdapter(VisionModelClientAdapter):
    provider_name = "Ollama"

    def __init__(self, model: str) -> None:
        import ollama

        self.model = model
        self.client = ollama.Client()

    def vision_task(
        self,
        image_base64: str,
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
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
        self.client.generate(model=self.model, prompt="", keep_alive=0)


_vision_model_client_adapters: dict[VisionModelProvider, VisionModelClientAdapter] = {}


def get_vision_model_client_adapter(
    provider: VisionModelProvider,
) -> VisionModelClientAdapter:
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
    """Cleans up a filename by removing unloved characters."""
    filename = filename.lower()
    filename = re.sub(r"^[^a-zA-Z_]+", "", filename)  # strip leading whitespace
    filename = re.sub(r"[\s_-]+", "_", filename)  # whitespace to underscore
    filename = re.sub(r"[^a-zA-Z0-9_.]", "", filename)  # strip special characters
    filename = re.sub(r"[\s_-]*\.+", ".", filename)  # whitespace before dot

    return filename


def fix_extension(current_filename: str, suggested_filename: str) -> str:
    """
    Ensures that the suggested filename has to correct extension, which
    is something GPT seems to struggle with. Renaming an image file to
    have the wrong extension will create a mismatch between the contents
    and extension, something we want to avoid.
    """
    current_ext = os.path.splitext(current_filename)[1].lower()
    suggested_base, suggested_ext = os.path.splitext(suggested_filename)
    if current_ext != suggested_ext:
        suggested_filename = suggested_base + current_ext
    return suggested_filename


def path_name_ext(path: str) -> tuple[str, str, str]:
    """
    Splits a full path name into a directory, base name,
    and extension, e.g. ("/static/images/", "logo", ".png")
    """
    dir_path = os.path.dirname(path)
    filename_with_ext = os.path.basename(path)
    filename, ext = os.path.splitext(filename_with_ext)
    if not dir_path.endswith(os.sep):
        dir_path += os.sep
    return (dir_path, filename, ext)


def scramble(filename: str) -> str:
    """Hashes a filename to obscure it. Only used for testing."""
    with TemporarySeed(seed=hash(filename)):
        return "".join(random.sample(string.ascii_letters, k=8))


def resize_image_to_fit(
    image: Image.Image | str, max_dimension: int = 512
) -> Image.Image:
    """
    Resizes the image to always fit within a 512x512 square
    regardless of aspect ratio. The returned image will always
    be smaller than 512 along both dimensions but will preserve
    its original aspect ratio. This allows it to consume only
    one "tile" in the GPT API.
    """
    # read from disk if given as filename
    if isinstance(image, str):
        image = Image.open(image)
    original_width, original_height = image.size

    # Determine which dimension is larger and calculate scaling factor
    if max(original_width, original_height) > max_dimension:
        if original_width > original_height:
            scaling_factor = max_dimension / original_width
        else:
            scaling_factor = max_dimension / original_height

        # Calculate new dimensions based on scaling factor
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # Resize the image to the new dimensions
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


def base64_encode_image(image: Image.Image | Pathish) -> str:
    """
    Encodes a Pillow image as base64 in a format GPT
    will accept.
    """

    if not isinstance(image, Image.Image):
        image = Image.open(os.fspath(image))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_data = buffer.getvalue()
    base64_encoded_bytes = base64.b64encode(byte_data)

    return base64_encoded_bytes.decode("utf-8")


def tag_image(
    filepath: str,
    client_adapter: VisionModelClientAdapter,
    prompt_template: str = IMAGE_PROMPT_TEMPLATE,
) -> dict[str, Any]:
    # handle local or remote images
    if filepath.startswith("http"):
        url = filepath
        filename = urlsplit(url).path.split("/")[-1]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = resize_image_to_fit(image)
    else:
        _, filename = os.path.split(filepath)
        image = resize_image_to_fit(filepath)

    base64_image_data = base64_encode_image(image)

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
    data["original_filepath"] = filepath
    data["original_filename"] = filename
    data["total_tokens"] = response.total_tokens
    data["provider_name"] = client_adapter.provider_name
    data["model"] = response.model
    data["width"] = image.size[0]
    data["height"] = image.size[1]
    data["vision_duration"] = vision_duration

    return data


def tag_images(
    filepaths: Iterable[str],
    output_filename: str | os.PathLike[str],
    retry_errors: bool = False,
    verbose: int = 1,
    provider: VisionModelProvider = VisionModelProvider.OPENAI,
    instructions_filename: str | os.PathLike[str] | None = None,
) -> None:
    client_adapter = get_vision_model_client_adapter(provider)
    if instructions_filename is None:
        prompt_template = IMAGE_PROMPT_TEMPLATE
    else:
        with open(instructions_filename, encoding="utf-8") as instructions_file:
            prompt_template = instructions_file.read()
    if verbose >= 1:
        print(f"Using {client_adapter}")
    file_already_exists = os.path.exists(output_filename)
    mode = "a" if file_already_exists else "w"

    try:
        with open(output_filename, mode, newline="", encoding="utf-8") as csvfile:
            columns = csv_columns
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            if not file_already_exists:
                writer.writeheader()

            vision_durations = []

            for index, filepath in enumerate(filepaths):
                row_start_time = time.perf_counter()

                try:
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
                        original_filename = os.path.basename(filepath)
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


def previously_tagged_filenames(metadata_filename: str | os.PathLike[str]) -> set[str]:
    if not os.path.exists(metadata_filename):
        return set()

    tagged_filenames = set()
    with open(metadata_filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("status") != "ok":
                continue

            for column in ["original_filename", "clean_filename"]:
                value = row.get(column)
                if not value:
                    continue
                tagged_filenames.add(os.path.basename(value))

    return tagged_filenames


def find_images(
    dirs: str | Iterable[str],
    max_days_old: float | None = None,
    metadata_filename: str | os.PathLike[str] | None = None,
    extension_filter: Iterable[str] | None = WELCOME_EXTENSIONS,
) -> list[str]:
    if max_days_old is None:
        max_days_old = float("Inf")

    if isinstance(dirs, str):
        dirs = [dirs]

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

    filepaths = []
    for dir in dirs:
        for filename in os.listdir(dir):
            filepath = os.path.join(dir, filename)
            if not os.path.isfile(filepath):
                continue
            if (current_time - os.path.getmtime(filepath)) >= max_days_old * 86400:
                continue
            if allowed_extensions is not None:
                extension = os.path.splitext(filepath)[1].lower()
                if extension not in allowed_extensions:
                    continue
            if filename in tagged_filenames:
                continue
            filepaths.append(filepath)

    return filepaths


def scramble_image_directory(
    input_dir: str,
    output_dir: str,
    max_dimension: int = 512,
) -> None:
    for filepath in find_images(input_dir):
        path, name, ext = path_name_ext(filepath)
        scrambled_name = scramble(name)
        new_filepath = os.path.join(output_dir, scrambled_name + ext)
        thumbnail = resize_image_to_fit(filepath, max_dimension)
        thumbnail.save(new_filepath)


def rename_images(
    csv_filename: str | os.PathLike[str],
    verbose: int = 1,
    dry_run: bool = False,
) -> None:
    metadata_df = pd.read_csv(csv_filename)
    metadata_updated = False
    display_directory = os.path.dirname(os.fspath(csv_filename)) or os.curdir
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for index, row in metadata_df.iterrows():
        source = row["original_filepath"]

        if row["status"] != "ok" or not row["clean_filename"]:
            if verbose >= 2:
                print(f"skipping errored row {index} {source!r}")
            continue

        # new filename
        directory, old_filename = os.path.split(source)
        new_filename = row["clean_filename"]
        target = os.path.join(directory, new_filename)

        # old filename
        if not os.path.isfile(source):
            if verbose >= 2:
                print(f"source file {source!r} is missing!")
            if verbose >= 1 and not os.path.isfile(target):
                print(f"both source file {source!r} and {target!r} are missing!")
            continue

        # check for no-op
        if target == source:
            if verbose >= 2:
                print(f"no rename necessary for {source!r}")
            continue

        # ensure extension matches
        source_ext = os.path.splitext(source)[1]
        target_base, target_ext = os.path.splitext(target)
        if source_ext.lower() != target_ext.lower():
            if verbose >= 1:
                print(
                    f"Mismatched file extensions between {source!r} and {target!r}; skipping rename!"
                )
            continue

        # check for name collisions
        if os.path.isfile(target):
            if verbose >= 1:
                print(f"target {target!r} already exists!")
            target = make_unique(target)
            if verbose >= 1:
                print(f"proceeding with target {target!r}.")

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
                os.rename(source, target)
                target_filename = os.path.basename(target)
                if row["clean_filename"] != target_filename:
                    metadata_df.at[index, "clean_filename"] = target_filename
                    metadata_updated = True
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            else:
                print(f"error renaming {source!r} to {target!r}!")
            traceback.print_exc()

    if metadata_updated:
        metadata_df.to_csv(csv_filename, index=False)


def shelve_images(
    csv_filename: str | os.PathLike[str],
    verbose: int = 1,
    dry_run: bool = False,
) -> None:
    metadata_df = pd.read_csv(csv_filename)
    upload_directory = os.path.dirname(os.fspath(csv_filename)) or os.curdir
    display_directory = os.path.dirname(upload_directory) or os.curdir
    if verbose == 1:
        print(f"working in {quote_display_path(display_directory)}")
    for index, row in metadata_df.iterrows():
        original_source = row["original_filepath"]

        if row["status"] != "ok" or not row["category"]:
            if verbose >= 2:
                print(f"skipping row {index} {original_source!r}")
            continue

        source_directory = os.path.dirname(original_source)
        source_filename = row["clean_filename"] or row["original_filename"]
        source = os.path.join(source_directory, source_filename)
        if not os.path.isfile(source):
            source = original_source

        if not os.path.isfile(source):
            if verbose >= 1:
                print(f"source file {source!r} is missing!")
            continue

        category = str(row["category"]).strip()
        target_directory = os.path.normpath(
            os.path.join(source_directory, os.pardir, category)
        )
        if not os.path.isdir(target_directory):
            if verbose >= 1:
                print(
                    f"target directory {target_directory!r} is missing; skipping {source!r}"
                )
            continue

        target = os.path.join(target_directory, os.path.basename(source))

        if target == source:
            if verbose >= 2:
                print(f"no move necessary for {source!r}")
            continue

        if os.path.isfile(target):
            if verbose >= 1:
                print(f"target {target!r} already exists!")
            target = make_unique(target)
            if verbose >= 1:
                print(f"proceeding with target {target!r}.")

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
                os.rename(source, target)
            if verbose >= 1:
                print("success!")
        except Exception:
            if verbose >= 1:
                print("error!")
            else:
                print(f"error moving {source!r} to {target!r}!")
            traceback.print_exc()


def generate_gallery(
    csv_filename: str | os.PathLike[str],
    output_filename: str | os.PathLike[str],
    verbose: int = 1,
) -> None:
    # read the metadata and prepare for merge
    metadata_df = pd.read_csv(csv_filename)
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
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(output)
