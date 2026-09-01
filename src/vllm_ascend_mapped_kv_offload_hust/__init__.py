"""Import-only mapped KV-offload contracts and migration descriptor."""

from .contract import (
    CanonicalKvDataRef,
    CanonicalTensorLayout,
    MappedGatherCapabilities,
    validate_mapped_gather_layout,
)

__all__ = [
    "CanonicalKvDataRef",
    "CanonicalTensorLayout",
    "MappedGatherCapabilities",
    "MappedKvOffloadDescriptor",
    "validate_mapped_gather_layout",
]


class MappedKvOffloadDescriptor:
    """Metadata anchor; importing the package installs no runtime hooks."""

    extension_id = "org.vllm-hust.ascend-mapped-kv-offload"
    activation_status = "import_only"
