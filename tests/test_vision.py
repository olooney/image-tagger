from typing import cast

from pydantic import BaseModel

from image_tagger import vision
from tests._workflow_split import export_tests

export_tests(globals(), "vision")


def test_get_vision_model_client_adapter_caches_by_provider(
    monkeypatch,
) -> None:
    """Return one cached adapter per provider."""
    created: list[str] = []

    class FakeOpenAIAdapter:
        def __init__(self) -> None:
            created.append("openai")

    class FakeOllamaAdapter:
        def __init__(self, model: str) -> None:
            created.append(model)

    monkeypatch.setattr(vision, "OpenAIVisionModelClientAdapter", FakeOpenAIAdapter)
    monkeypatch.setattr(vision, "OllamaVisionModelClientAdapter", FakeOllamaAdapter)
    monkeypatch.setattr(vision, "_vision_model_client_adapters", {})

    openai_first = vision.get_vision_model_client_adapter(
        vision.VisionModelProvider.OPENAI
    )
    openai_second = vision.get_vision_model_client_adapter(
        vision.VisionModelProvider.OPENAI
    )
    gemma = vision.get_vision_model_client_adapter(vision.VisionModelProvider.GEMMA)
    qwen = vision.get_vision_model_client_adapter(vision.VisionModelProvider.QWEN)

    assert openai_first is openai_second
    assert created.count("openai") == 1
    assert created.count(vision.GEMMA_MODEL) == 1
    assert created.count(vision.QWEN_MODEL) == 1
    assert isinstance(gemma, FakeOllamaAdapter)
    assert isinstance(qwen, FakeOllamaAdapter)


def test_get_vision_model_client_adapter_rejects_unknown_provider() -> None:
    """Raise when provider value is unsupported."""
    try:
        vision.get_vision_model_client_adapter(
            cast("vision.VisionModelProvider", "not-a-provider"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid provider.")


def test_ollama_vision_task_allows_long_responses() -> None:
    """Request enough output tokens for long structured responses."""
    calls: list[dict[str, object]] = []

    class ResponseData(BaseModel):
        value: str

    class FakeClient:
        def chat(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "message": {"content": '{"value":"ok"}'},
                "model": "demo-model",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }

    adapter = vision.OllamaVisionModelClientAdapter.__new__(
        vision.OllamaVisionModelClientAdapter,
    )
    adapter.model = "demo-model"
    adapter.client = FakeClient()

    result = adapter.vision_task("image-data", "prompt", ResponseData)

    assert calls[0]["options"] == {
        "temperature": 0,
        "image_min_tokens": 1120,
        "image_max_tokens": 1120,
        "num_ctx": 16384,
        "num_predict": 8192,
    }
    assert result.data == ResponseData(value="ok")


def test_ollama_cleanup_unloads_model() -> None:
    """Unload the model with an empty prompt and zero keep_alive."""
    calls: list[tuple[str, str, int]] = []

    class FakeClient:
        def generate(self, *, model: str, prompt: str, keep_alive: int) -> None:
            calls.append((model, prompt, keep_alive))

    adapter = vision.OllamaVisionModelClientAdapter.__new__(
        vision.OllamaVisionModelClientAdapter,
    )
    adapter.model = "demo-model"
    adapter.client = FakeClient()

    adapter.cleanup()

    assert calls == [("demo-model", "", 0)]
