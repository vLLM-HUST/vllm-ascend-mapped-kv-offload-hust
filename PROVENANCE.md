# Provenance

Immutable source archives:

- [intellistream/vllm-ascend-hust-legacy-20260831](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831)
- [intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831)

Primary history:

- [Ascend PR #52: initial mapped-host KV cache gather](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/52)
- [Ascend PR #67: device KV transfer prototype](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/67)
- [Ascend PR #153: mapped gather operator, lifecycle, offload spec, tests, and evidence](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/153)
- [Core PR #150: async KV-load scheduler wakeup dependency](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/150)

PR #153 credits Wangjie for material mainline-port work. Extraction must
preserve per-file authorship and must not assign that work solely from the PR
owner.

Migrated from PR #153:

| Legacy source | New path | Scope |
| --- | --- | --- |
| `vllm_ascend/kv_offload/experimental_mapped.py::_validate_mapped_gather_layout` | `src/vllm_ascend_mapped_kv_offload_hust/contract.py` | Dependency-neutral representation of the exact nonempty, paired, contiguous, unpadded, 32-byte-aligned and canonical-page checks |
| `_resolve_mapped_gather_ops` runtime checks | `MappedGatherCapabilities` in the same module | Fail-closed all-symbol/runtime-ready capability contract |

No Worker adapter, torch integration, C++ operator, build glue, benchmark
result, or runtime activation has been migrated. Those retain the original
per-file authorship and require a public host contract plus real Ascend
validation before extraction.
