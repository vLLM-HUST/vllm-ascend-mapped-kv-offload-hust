import json
from pathlib import Path


def test_migration_manifest_is_import_only() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "vllm_ascend_mapped_kv_offload_hust"
        / "vllm-hust-extension-v0.2.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["extension_id"] == "org.vllm-hust.ascend-mapped-kv-offload"
    assert value["host"]["name"] == "vllm-ascend"
    assert value["implementation"][0]["status"] == "import_only"
    assert value["activation"]["entry_points"] == []
