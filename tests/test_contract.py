import pytest

from vllm_ascend_mapped_kv_offload_hust.contract import (
    CanonicalKvDataRef,
    CanonicalTensorLayout,
    MappedGatherCapabilities,
    validate_mapped_gather_layout,
)


def layout(row_bytes: int = 256) -> CanonicalTensorLayout:
    return CanonicalTensorLayout(row_bytes, row_bytes)


def test_accepts_exact_contiguous_aligned_layout() -> None:
    validate_mapped_gather_layout(
        (layout(),),
        (layout(),),
        ((CanonicalKvDataRef(0, 256),),),
    )


@pytest.mark.parametrize(
    ("npu", "cpu", "groups", "message"),
    [
        ((), (), (), "empty or mismatched"),
        ((layout(),), (layout(), layout()), (), "empty or mismatched"),
        (
            (CanonicalTensorLayout(256, 512),),
            (layout(),),
            (),
            "padded/strided",
        ),
        ((layout(48),), (layout(48),), (), "32-byte aligned"),
        ((layout(),), (layout(),), ((CanonicalKvDataRef(1, 256),),), "missing"),
        (
            (layout(),),
            (layout(),),
            ((CanonicalKvDataRef(0, 128),),),
            "canonical row size",
        ),
    ],
)
def test_rejects_unsupported_layouts(npu, cpu, groups, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_mapped_gather_layout(npu, cpu, groups)


def test_native_capability_is_indivisible() -> None:
    MappedGatherCapabilities(True, True, True, True).require_ready()

    with pytest.raises(RuntimeError, match="unregister_host_pool"):
        MappedGatherCapabilities(True, True, False, True).require_ready()
