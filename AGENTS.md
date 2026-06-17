# Agent Notes

- Prefer focused changes in `src/cli.py`, `src/image_tagger.py`, and `tests/test_cli_workflow.py`; this project keeps most workflow coverage in one end-to-end test file.
- Default CLI verbosity is `1`; preserve quiet/progress output behavior when adding workflow features.
- For command-line display of filenames, use `quote_display_path()` so paths are quoted only when needed.
- Do not move dev tools such as `pytest`, `pre-commit`, or `ty` out of runtime dependencies unless explicitly requested.
