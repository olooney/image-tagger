from collections.abc import Callable

from tests import workflow_cases as _cases


def _category(name: str) -> str:
    lowered = name.casefold()
    if "dedupe" in lowered:
        return "dedupe"
    if "gallery" in lowered:
        return "gallery"
    if "wall" in lowered or "aspect_ratio" in lowered:
        return "wall"
    if any(token in lowered for token in ["tag", "quad", "image_tag"]):
        return "tagging"
    if any(
        token in lowered
        for token in [
            "transform",
            "crop",
            "hough",
            "contour",
            "perspective",
            "vlm",
            "provider",
        ]
    ):
        return "vision"
    if any(token in lowered for token in ["clip", "embed", "comparison", "same_image"]):
        return "compare"
    return "core"


def export_tests(namespace: dict[str, object], category: str) -> None:
    tests: dict[str, Callable[..., object]] = {
        name: obj
        for name, obj in vars(_cases).items()
        if name.startswith("test_") and callable(obj)
    }
    selected = [name for name in sorted(tests) if _category(name) == category]
    for name in selected:
        namespace[name] = tests[name]

    # Re-export fixtures so imported tests can resolve local fixture names.
    for name, obj in vars(_cases).items():
        if callable(obj) and (
            hasattr(obj, "_pytestfixturefunction")
            or hasattr(obj, "_fixture_function_marker")
        ):
            namespace[name] = obj
