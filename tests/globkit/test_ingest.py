from src.globkit.ingest import (
    load_csv_dataset,
    load_data_files,
    load_time_series_data,
)


def test_loaders(tmp_path):
    (tmp_path / "a.csv").write_text("x,ts\n1,2024-01-01\n")
    (tmp_path / "b.csv").write_text("x,ts\n2,2024-01-02\n")
    df = load_csv_dataset(str(tmp_path / "*.csv"))
    assert len(df) == 2
    dfts = load_time_series_data(str(tmp_path / "*.csv"), date_column="ts")
    assert list(dfts["x"]) == [1, 2]
    mixed = load_data_files([str(tmp_path / "*.csv")])
    assert mixed and next(iter(mixed.values())).equals(df[df.columns])
