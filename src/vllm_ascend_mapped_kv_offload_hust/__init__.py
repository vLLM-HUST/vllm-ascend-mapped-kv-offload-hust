"""Import-only descriptor for the owner-led mapped KV-offload migration."""


class MappedKvOffloadDescriptor:
    """Metadata anchor; no runtime hooks are installed by this package."""

    extension_id = "org.vllm-hust.ascend-mapped-kv-offload"
    activation_status = "import_only"
