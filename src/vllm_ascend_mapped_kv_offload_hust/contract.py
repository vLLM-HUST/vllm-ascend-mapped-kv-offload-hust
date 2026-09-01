# SPDX-License-Identifier: Apache-2.0
"""Dependency-neutral mapped-host KV layout and capability contract.

This is the host-independent portion of the fail-closed checks implemented by
legacy Ascend PR #153. It deliberately models facts rather than torch tensors
or vLLM private classes, so a host provider can validate a plan before loading
the native operator or importing a vLLM worker.
"""

from __future__ import annotations

from dataclasses import dataclass

MAPPED_GATHER_ALIGNMENT_BYTES = 32


@dataclass(frozen=True)
class CanonicalTensorLayout:
    """Canonical flattened KV tensor row geometry."""

    row_bytes: int
    row_stride_bytes: int
    contiguous: bool = True

    def __post_init__(self) -> None:
        if self.row_bytes <= 0:
            raise ValueError("row_bytes must be positive")
        if self.row_stride_bytes <= 0:
            raise ValueError("row_stride_bytes must be positive")


@dataclass(frozen=True)
class CanonicalKvDataRef:
    """One cache-group reference into the canonical tensor list."""

    tensor_index: int
    page_size_bytes: int

    def __post_init__(self) -> None:
        if self.tensor_index < 0:
            raise ValueError("tensor_index must be nonnegative")
        if self.page_size_bytes <= 0:
            raise ValueError("page_size_bytes must be positive")


@dataclass(frozen=True)
class MappedGatherCapabilities:
    """Native symbols required as one indivisible mapped-gather capability."""

    gather: bool
    register_host_pool: bool
    unregister_host_pool: bool
    runtime_ready: bool

    def require_ready(self) -> None:
        missing = [
            name
            for name, available in (
                ("gather", self.gather),
                ("register_host_pool", self.register_host_pool),
                ("unregister_host_pool", self.unregister_host_pool),
                ("runtime_ready", self.runtime_ready),
            )
            if not available
        ]
        if missing:
            raise RuntimeError(
                "mapped gather capability is incomplete: " + ", ".join(missing)
            )


def validate_mapped_gather_layout(
    npu_tensors: tuple[CanonicalTensorLayout, ...],
    cpu_tensors: tuple[CanonicalTensorLayout, ...],
    cache_groups: tuple[tuple[CanonicalKvDataRef, ...], ...],
) -> None:
    """Reject layouts outside the intentionally narrow legacy domain."""
    if not npu_tensors or len(npu_tensors) != len(cpu_tensors):
        raise ValueError("canonical CPU/NPU tensor lists are empty or mismatched")

    for tensor_index, (cpu_tensor, npu_tensor) in enumerate(
        zip(cpu_tensors, npu_tensors, strict=True)
    ):
        if not cpu_tensor.contiguous or not npu_tensor.contiguous:
            raise ValueError(f"canonical tensor {tensor_index} is not contiguous")
        if (
            cpu_tensor.row_stride_bytes != cpu_tensor.row_bytes
            or npu_tensor.row_stride_bytes != npu_tensor.row_bytes
        ):
            raise ValueError(
                f"canonical tensor {tensor_index} has a padded/strided row"
            )
        if npu_tensor.row_bytes % MAPPED_GATHER_ALIGNMENT_BYTES:
            raise ValueError(
                f"canonical tensor {tensor_index} page size "
                f"{npu_tensor.row_bytes} is not "
                f"{MAPPED_GATHER_ALIGNMENT_BYTES}-byte aligned"
            )

    for group_index, group in enumerate(cache_groups):
        for data_ref in group:
            if data_ref.tensor_index >= len(npu_tensors):
                raise ValueError(
                    f"KV group {group_index} references missing tensor "
                    f"{data_ref.tensor_index}"
                )
            row_bytes = npu_tensors[data_ref.tensor_index].row_bytes
            if data_ref.page_size_bytes != row_bytes:
                raise ValueError(
                    f"KV group {group_index} has unpadded page size "
                    f"{data_ref.page_size_bytes}, but canonical row size is "
                    f"{row_bytes}"
                )
