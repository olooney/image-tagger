from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from .constants import GEMMA_MODEL, OPENAI_MODEL, QWEN_MODEL
from .util import connect_to_openai


class VisionModelProvider(Enum):
    """Supported vision model providers."""

    OPENAI = "openai"
    GEMMA = "gemma"
    QWEN = "qwen"


@dataclass(frozen=True)
class VisionTaskResult:
    """Validated structured vision model response data."""

    data: BaseModel
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
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run a vision task."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Release provider resources."""
        raise NotImplementedError


class OpenAIVisionModelClientAdapter(VisionModelClientAdapter):
    """OpenAI vision provider adapter."""

    provider_name = "OpenAI"

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        """Create an OpenAI adapter."""
        self.model = model
        self.client = connect_to_openai()

    def vision_task(
        self,
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run an OpenAI vision request."""
        image_base64_values = (
            [image_base64] if isinstance(image_base64, str) else image_base64
        )
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_value}"},
            }
            for image_value in image_base64_values
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            response_format=response_format,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_content,
                    ],
                },
            ],
        )
        data = response.choices[0].message.parsed
        if data is None:
            raise ValueError(
                "OpenAI response did not match the requested response model."
            )
        total_tokens = response.usage.total_tokens if response.usage is not None else 0
        return VisionTaskResult(
            data=data,
            model=response.model,
            total_tokens=total_tokens,
        )

    def cleanup(self) -> None:
        """OpenAI cleanup hook."""


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
        image_base64: str | list[str],
        prompt: str,
        response_format: type[BaseModel],
    ) -> VisionTaskResult:
        """Run an Ollama vision request."""
        image_base64_values = (
            [image_base64] if isinstance(image_base64, str) else image_base64
        )
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_base64_values,
                },
            ],
            format=response_format.model_json_schema(),
            options={
                "temperature": 0,
                "image_min_tokens": 1120,
                "image_max_tokens": 1120,
                "num_ctx": 16384,
                "num_predict": 8192,
            },
        )
        message = response.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("Ollama response did not include JSON content.")
        return VisionTaskResult(
            data=response_format.model_validate_json(content),
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
                GEMMA_MODEL,
            )
        elif provider == VisionModelProvider.QWEN:
            _vision_model_client_adapters[provider] = OllamaVisionModelClientAdapter(
                QWEN_MODEL,
            )
        else:
            raise ValueError(f"Unsupported vision model provider: {provider.value}")
    return _vision_model_client_adapters[provider]
