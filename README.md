# Ascend Mapped-host KV Offload

Owner-maintained extraction of the mapped-host KV gather and offload work
preserved in the archived vLLM-HUST repositories.

**Status: installable contract package (`import_only`). This repository is
discoverable by the Extension Manager but is not yet runnable.**

The intended extension supplies an opt-in Ascend KV-offload backend with
address-stable host-memory mapping, a packaged gather operator, explicit lease
lifecycle, and fail-closed capability checks. Native CPU offloading remains the
host default.

The package now contains the dependency-neutral layout and native-capability
checks extracted from legacy PR #153. It intentionally does not contain the
vLLM Worker adapter or native operator yet: those require a reviewed public
host seam and a real Ascend build/test receipt. Importing the package does not
load torch, vLLM, vLLM Ascend, or a native library.

```bash
pip install -e .
vllm-hust-ext extension inspect org.vllm-hust.ascend-mapped-kv-offload
```

Enablement is intentionally rejected until the owner approves the extracted
implementation, the host contract, and a real-device acceptance receipt.

See [PROVENANCE.md](PROVENANCE.md) and [MAINTAINERS.md](MAINTAINERS.md).
