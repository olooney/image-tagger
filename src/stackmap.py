"""Library shelf configuration loading and discovery."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


STACKMAP_FILENAME: str = ".stackmap"
SHELF_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class StackMap:
    """Resolved library shelf directories from a `.stackmap` file."""

    filename: Path
    shelves: dict[str, Path]
    shelf_descriptions: dict[str, str]

    @classmethod
    def load(cls, filename: Path) -> "StackMap":
        """Load and resolve a stack map file."""
        resolved_filename = filename.resolve()
        try:
            raw_contents = resolved_filename.read_text(encoding="utf-8")
            contents: Any = yaml.safe_load(raw_contents)
        except OSError as error:
            raise ValueError(f"could not read stack map {resolved_filename}: {error}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"invalid stack map {resolved_filename}: {error}") from error

        if not isinstance(contents, dict):
            raise ValueError(f"stack map {resolved_filename} must contain shelf mappings.")

        document = yaml.compose(raw_contents)
        descriptions_by_line = _inline_comments_by_value_line(document, raw_contents)

        shelves: dict[str, Path] = {}
        shelf_descriptions: dict[str, str] = {}
        for alias, directory in contents.items():
            if not isinstance(alias, str) or not SHELF_NAME_PATTERN.fullmatch(alias):
                raise ValueError(
                    "stack map shelf names must be single identifiers containing only "
                    "letters, digits, and underscores."
                )
            if not isinstance(directory, str) or not directory:
                raise ValueError(f"stack map shelf {alias!r} must have a directory path.")
            path = Path(directory)
            shelves[alias] = (
                path.resolve() if path.is_absolute() else (resolved_filename.parent / path).resolve()
            )

            if description := descriptions_by_line.get(len(shelves) - 1):
                shelf_descriptions[alias] = description

        if "default" not in shelves:
            raise ValueError(f"stack map {resolved_filename} must define a default shelf.")
        return cls(
            filename=resolved_filename,
            shelves=shelves,
            shelf_descriptions=shelf_descriptions,
        )

    @property
    def default_directory(self) -> Path:
        """Return the directory used when no command directory is supplied."""
        return self.shelves["default"]

    @property
    def categories(self) -> list[str]:
        """Return shelves available for vision model categorization."""
        return [alias for alias in self.shelves if alias != "default"]

    @property
    def category_descriptions(self) -> dict[str, str]:
        """Return optional prompt guidance for categorization shelves."""
        return {
            alias: description
            for alias, description in self.shelf_descriptions.items()
            if alias != "default"
        }

    def directory_for(self, alias: str) -> Path | None:
        """Return a shelf directory, if the alias is configured."""
        return self.shelves.get(alias)


def _inline_comments_by_value_line(
    document: yaml.Node | None,
    contents: str,
) -> dict[int, str]:
    """Extract YAML inline comments indexed by mapping-entry order."""
    if not isinstance(document, yaml.MappingNode):
        return {}

    descriptions: dict[int, str] = {}
    lines = contents.splitlines()
    for index, (_, value_node) in enumerate(document.value):
        line = lines[value_node.end_mark.line]
        suffix = line[value_node.end_mark.column :].lstrip()
        if suffix.startswith("#") and (description := suffix[1:].strip()):
            descriptions[index] = description

    return descriptions


def find_stackmap(start_directory: Path | None = None) -> Path | None:
    """Find `.stackmap` from the working directory upward, then home."""
    start = (start_directory or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    home = Path.home().resolve()
    if home not in candidates:
        candidates.append(home)
    for directory in candidates:
        filename = directory / STACKMAP_FILENAME
        if filename.is_file():
            return filename
    return None