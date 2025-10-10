from src.globkit.ml import prepare_ml_dataset


def test_prepare_ml_dataset(tmp_path):
    (tmp_path / "part1.csv").write_text("a,b,target\n1,2,0\n")
    (tmp_path / "part2.csv").write_text("a,b,target\n3,4,1\n")
    X, y = prepare_ml_dataset(str(tmp_path / "*.csv"))
    assert list(X.columns) == ["a", "b"]
    assert list(y) == [0, 1]
