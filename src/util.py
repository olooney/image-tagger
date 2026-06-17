import logging
import os
import random
import subprocess
import sys
import webbrowser
from collections.abc import Iterable
from datetime import datetime
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import tenacity
import yaml

logger: logging.Logger = logging.getLogger(__name__)

type Pathish = PathLike | str


def human_join(
    items: Iterable[str],
    conjunction: str = "and",
) -> str:
    """Join words with a final conjunction."""
    items = list(items)  # needed for items[:-1]

    if len(items) > 2:
        return ", ".join(items[:-1]) + ", " + conjunction + " " + items[-1]
    elif len(items) == 2:
        return (" " + conjunction + " ").join(items)
    elif items:
        return items[0]
    else:
        return ""


def now(with_time: bool = True) -> str:
    """Return the current date or datetime string."""
    current_datetime = datetime.now()
    format = "%Y-%m-%d"
    if with_time:
        format += " %H:%M"
    return current_datetime.strftime(format)


def preview(html_path: Pathish) -> None:
    """Open an HTML file in VS Code or a browser."""
    html_uri = Path(html_path).resolve().as_uri()
    try:
        subprocess.run(
            ["code", "--reuse-window", "--open-url", html_uri],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        webbrowser.open(html_uri)


def make_unique(path: Pathish) -> str:
    """Return an available path by adding suffixes."""
    path = Path(path)
    if not path.exists():
        return os.fspath(path)

    separator = "_" if path.stem[-1:].isdigit() else ""
    for suffix in range(2, 10):
        candidate = path.with_name(f"{path.stem}{separator}{suffix}{path.suffix}")
        if not candidate.exists():
            return os.fspath(candidate)

    raise FileExistsError(f"No available filename from {path!r} through suffix 9.")


class TemporarySeed:
    """Temporarily seed Python's random generator."""

    def __init__(self, seed: int | None = None) -> None:
        """Store the seed for later use."""
        self.seed = seed
        self.state: tuple[Any, ...] | None = None

    def __enter__(self) -> Self:
        """Seed the random generator."""
        # save the current random state before reseeding
        self.state = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the previous random state."""
        # restore the saved random state
        if self.state is None:
            return
        random.setstate(self.state)


class Config:
    """Simple object-backed config container."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        """Set config values as attributes."""
        if values:
            for key, value in values.items():
                setattr(self, key, value)

    @classmethod
    def load(cls: type[Self], filename: Pathish) -> Self:
        """Load YAML config from disk."""
        with open(filename, encoding="utf-8") as file:
            data = yaml.safe_load(file)

            credentials = cls()
            for key, value in data.items():
                setattr(credentials, key, value)
            return credentials

    def __repr__(self) -> str:
        """Represent public config values."""
        return f"Config({self.__dict__!r})"

    __str__ = __repr__


class Credentials(Config):
    """Config container that redacts secrets."""

    def __repr__(self) -> str:
        """Represent credentials with secrets hidden."""
        public: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if (
                "password" in key.lower()
                or "key" in key.lower()
                or "token" in key.lower()
            ):
                value = "********"
            public[key] = value
        return f"Config({public!r})"

    __str__ = __repr__


def total_size(
    obj: Any,
    seen: set[int] | None = None,
) -> int:
    """Recursively find an object's memory size."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    # use ndarray byte counts instead of object header size
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    else:
        size = sys.getsizeof(obj, 0)

    if isinstance(obj, dict):
        size += sum(total_size(k, seen) + total_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, np.ndarray)):
        size += sum(total_size(i, seen) for i in obj)

    return size


def connect_to_openai() -> Any:
    """Create an OpenAI client using local credentials."""
    import openai

    openai_credentials_filename = Path.home() / ".openai" / "credentials.yaml"
    Credentials.load(openai_credentials_filename)
    client = openai.OpenAI()
    return client


retry_decorator: Any = tenacity.retry(
    wait=tenacity.wait_exponential(min=0.1, max=2),
    stop=tenacity.stop_after_attempt(3),  # because 4 is too many and 2 isn't enough.
    after=tenacity.after_log(logger, logging.ERROR),
    reraise=True,
)
