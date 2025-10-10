from src.globkit.patterns import iglob_iter, list_files


def test_basic_list_and_iter(tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    (tmp_path / "b.csv").write_text("y\n")
    got = list_files(str(tmp_path / "*.csv"))
    assert len(got) == 2
    got_iter = list(iglob_iter(str(tmp_path / "*.csv")))
    assert set(got) == set(got_iter)
