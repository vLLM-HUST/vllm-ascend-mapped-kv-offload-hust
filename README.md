# Ascend Mapped-host KV Offload

Owner-maintained extraction of the mapped-host KV gather and offload work
preserved in the archived vLLM-HUST repositories.

**Status: migration scaffold (`import_only`). This repository is discoverable
by the vLLM-HUST Extension Manager but is not yet runnable.**

The intended extension supplies an opt-in Ascend KV-offload backend with
address-stable host-memory mapping, a packaged gather operator, explicit lease
lifecycle, and fail-closed capability checks. Native CPU offloading remains the
host default.

The extraction must retain the original NPU correctness, lifecycle, packaging,
and scoped performance evidence. It must use a narrow host-owned KV-offload
contract rather than patching vLLM or vLLM Ascend at import time.

```bash
pip install -e .
vllm-hust-ext extension inspect org.vllm-hust.ascend-mapped-kv-offload
```

Enablement is intentionally rejected until the owner approves the extracted
implementation, the host contract, and a real-device acceptance receipt.

See [PROVENANCE.md](PROVENANCE.md) and [MAINTAINERS.md](MAINTAINERS.md).
