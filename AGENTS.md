# Agent Notes

- Default CLI verbosity is `1`; preserve quiet/progress output behavior when adding workflow features.
- CLI handlers receive the final `args.verbose` integer after `-v`/`-q` deltas are resolved; pass it explicitly through long-running functions rather than reading CLI globals.
- For command-line display of filenames, use `quote_display_path()` so paths are quoted only when needed.
- For file operations, print the operation with ` ...` and no newline before running it, then print `success!` or `error!` after.
- Keep Python functions and methods annotated with terse necessary docstrings; annotate module globals when adding them.
- Prefer `typing.Literal` over `Enum` for constrained string fields.
- Add a matching `justfile` recipe whenever adding a CLI command.
- For multiline function signatures and long literals, use trailing commas and one item or argument per line.
- Add short one-line comments before non-obvious code blocks, not at the end of code lines.
- Do not move dev tools such as `pytest`, `pre-commit`, or `ty` out of runtime dependencies unless explicitly requested.
- Always use `uv` for Python dependency and environment operations so `uv.lock` stays in sync with `pyproject.toml`.
- NEVER touch or modify the user's `.stackmap` config file. You may modify `tests/.stackmap` when writing tests.
