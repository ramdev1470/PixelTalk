from .multimodal_vlm import PixelTalkMultimodalVLM
from .safety_guardrail import MultimodalSafetyGuardrail

__all__ = [
    "PixelTalkMultimodalVLM",
    "MultimodalSafetyGuardrail",
    "split_dataset_by_image_groups",
    "MultimodalVLDataset",
    "DATA_RECORDS",
]


def __getattr__(name):
    # Training-only helpers need torch; import them lazily so the app and tests
    # that only need the chat pipeline don't pay for (or require) them.
    if name in ("split_dataset_by_image_groups", "MultimodalVLDataset", "DATA_RECORDS"):
        from . import grouped_dataset
        return getattr(grouped_dataset, name)
    raise AttributeError(f"module 'multimodal' has no attribute {name!r}")
