import sys
import os
import tenacity
from collections.abc import Iterable
import numpy as np
import yaml
import random
import subprocess
import webbrowser
from datetime import datetime
import logging
from pathlib import Path
from os import PathLike

logger = logging.getLogger(__name__)

Pathish = PathLike | str


def human_join(items: Iterable, conjunction="and"):
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
    current_datetime = datetime.now()
    format = "%Y-%m-%d"
    if with_time:
        format += " %H:%M"
    return current_datetime.strftime(format)


def preview(html_path: Pathish):
    html_uri = Path(html_path).resolve().as_uri()
    try:
        subprocess.run(
            ["code", "--reuse-window", "--open-url", html_uri],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        webbrowser.open(html_uri)


def make_unique(path: Pathish) -> str:
    path = os.fspath(path)
    if not os.path.exists(path):
        return path

    path_base, path_ext = os.path.splitext(path)
    separator = "_" if path_base[-1:].isdigit() else ""
    for suffix in range(2, 10):
        candidate = f"{path_base}{separator}{suffix}{path_ext}"
        if not os.path.exists(candidate):
            return candidate

    raise FileExistsError(f"No available filename from {path!r} through suffix 9.")


class TemporarySeed:
    """Context manager that temporarily seeds python's
    internal random number generator to a specific value,
    then restores it to its original state.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def __enter__(self):
        # Save the current random state
        self.state = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the saved random state
        random.setstate(self.state)


class Config:
    def __init__(self, values=None):
        if values:
            for key, value in values.items():
                setattr(self, key, value)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

            credentials = cls()
            for key, value in data.items():
                setattr(credentials, key, value)
            return credentials

    def __repr__(self):
        return f"Config({self.__dict__!r})"

    __str__ = __repr__


class Credentials(Config):
    def __repr__(self):
        public = {}
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


def total_size(obj, seen=None):
    """Recursively find the total memory size of a Python object."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    # Check if the object is a numpy array
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    else:
        size = sys.getsizeof(obj, 0)

    if isinstance(obj, dict):
        size += sum(total_size(k, seen) + total_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, np.ndarray)):
        size += sum(total_size(i, seen) for i in obj)

    return size


def connect_to_openai():
    import openai

    openai_credentials_filename = os.path.join(
        os.path.expanduser("~"), ".openai", "credentials.yaml"
    )
    Credentials.load(openai_credentials_filename)
    client = openai.OpenAI()
    return client


retry_decorator = tenacity.retry(
    wait=tenacity.wait_exponential(min=0.1, max=2),
    stop=tenacity.stop_after_attempt(3),  # because 4 is too many and 2 isn't enough.
    after=tenacity.after_log(logger, logging.ERROR),
    reraise=True,
)
