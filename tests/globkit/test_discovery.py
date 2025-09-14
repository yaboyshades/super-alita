from src.globkit.discovery import (
    analyze_file_collection,
    discover_data_files,
    optimize_file_loading,
)


def test_analyze(tmp_path):
    p = tmp_path/"sub"
    p.mkdir()
    f = p/"data.csv"
    f.write_text("a,b\n1,2\n")
    df = analyze_file_collection(str(tmp_path/"**/*.csv"))
    assert (df["path"] == str(f)).any()
    assert "size" in df.columns


def test_discover_and_batch(tmp_path):
    for i in range(12):
        (tmp_path/f"file_{i}.json").write_text("{}")
    found = discover_data_files(str(tmp_path), patterns=["*.json"])
    assert "*.json" in found and len(found["*.json"]) == 12
    batches = list(optimize_file_loading(found["*.json"], batch_size=5))
    assert [len(b) for b in batches] == [5, 5, 2]
